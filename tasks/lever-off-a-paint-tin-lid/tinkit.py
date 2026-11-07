"""tinkit: compound rigid bodies, a press-fit lid seal and tilted-grip waypoints for a yam_arena task file.

Written for the paint-tin task in this folder. Importing it is cheap (NumPy only): everything that needs Isaac
Sim is imported lazily, inside hooks that run once the simulator has started.

What it adds to a plain task file:

- :class:`Body`: one rigid body (or a fixed fixture, without a mass) made of several boxes and cylinders, each
  placed and turned freely in the body's frame. It subclasses ``task_file.Box`` only so that the task-file
  loader accepts it in ``OBJECTS``; ``size`` is its bounding box and ``pos`` is the body's origin.
- :class:`Seal`: the press-fit of a lid in a tin's rim, as ``points`` spring-free catches around the lid's edge.
  Each catch pulls the lid down onto the rim with a constant ``hold`` force (and pushes the tin up by the same
  amount) until that part of the lid edge has been raised ``release`` above its seat; then it lets go for the
  rest of the trial. Tilting the lid ``pop_tilt`` against the tin frees every remaining catch. The forces are
  applied before every physics substep (240 Hz), so a quick motion cannot skip them.
- :func:`tilted`: a waypoint orientation for ``yam_arena.demo`` that is not a straight-down grip, so a scripted
  demo can hold a tool at an angle.
- :func:`on_reset`: functions to run at every episode reset (before the objects settle), e.g. to clear latched
  violations.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import math
import sys
from dataclasses import dataclass, field

import numpy as np

from yam_arena import demo as _demo
from yam_arena.task_file import Box, ObjectState

__all__ = ["Shape", "box", "cyl", "ring", "prism12", "Body", "body", "BodyState", "Seal", "tilted", "on_reset", "finger_points", "install"]

Color = tuple[float, float, float]
DEBUG = bool(__import__("os").environ.get("TINKIT_DEBUG"))  # print seal releases as they happen


# -- geometry ---------------------------------------------------------------------------------------------------


def quat_to_matrix(q_xyzw) -> np.ndarray:
    x, y, z, w = (float(v) for v in q_xyzw)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def yaw_quat(yaw: float) -> tuple[float, float, float, float]:
    return (0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2))


@dataclass(frozen=True)
class Shape:
    """A collision shape in its body's frame: a box (``size`` = x, y, z edges) or a cylinder (``size`` = radius,
    height, along ``axis``), centred at ``centre`` and turned by ``yaw`` about the body's z axis."""

    kind: str
    centre: tuple[float, float, float]
    size: tuple[float, ...]
    color: Color
    axis: str = "Z"
    yaw: float = 0.0

    def half_extent(self) -> np.ndarray:
        """Half the shape's axis-aligned extent in the body frame."""
        if self.kind == "box":
            hx, hy, hz = (s / 2 for s in self.size)
            c, s = abs(math.cos(self.yaw)), abs(math.sin(self.yaw))
            return np.array([c * hx + s * hy, s * hx + c * hy, hz])
        r, h = self.size
        local = {"X": (h / 2, r, r), "Y": (r, h / 2, r), "Z": (r, r, h / 2)}[self.axis]
        hx, hy, hz = local
        c, s = abs(math.cos(self.yaw)), abs(math.sin(self.yaw))
        return np.array([c * hx + s * hy, s * hx + c * hy, hz])


def box(centre, size, color, yaw: float = 0.0) -> Shape:
    return Shape("box", tuple(map(float, centre)), tuple(map(float, size)), tuple(color), yaw=float(yaw))


def cyl(centre, radius, height, color, axis: str = "Z", yaw: float = 0.0) -> Shape:
    return Shape("cyl", tuple(map(float, centre)), (float(radius), float(height)), tuple(color), axis, float(yaw))


def ring(z0, z1, r_in, r_out, color, segments: int = 24) -> list[Shape]:
    """An upright ring (inner radius ``r_in``, outer ``r_out``, from height ``z0`` to ``z1`` in the body frame) as
    ``segments`` boxes, each just long enough to close the gaps at the outer face."""
    r_mid, width, h = (r_in + r_out) / 2, r_out - r_in, z1 - z0
    chord = 2 * r_out * math.tan(math.pi / segments) + 0.0003
    return [box((r_mid * math.cos(a), r_mid * math.sin(a), (z0 + z1) / 2), (chord, width, h), color, yaw=a + math.pi / 2)
            for a in (2 * math.pi * k / segments for k in range(segments))]


def prism12(centre, inradius, height, color) -> list[Shape]:
    """An upright regular 12-sided prism (``inradius`` to its flat sides) as six boxes turned 30 degrees apart: each
    box spans two opposite sides and is one side wide, and together they fill the 12-gon exactly.

    WHY (2026-11-07, sim trace): PhysX's custom cylinder geometry misses edge contacts between two cylinders: a
    2 mm lid disc tilted a few degrees on a cylindrical seat sank 6 mm into it. A lid cylinder on box seats holds.
    (Turning on /physics/collisionApproximateCylinders instead made every cylinder creep into its support.)"""
    side = 2 * inradius * math.tan(math.pi / 12)
    return [box(centre, (2 * inradius, side, height), color, yaw=k * math.pi / 6) for k in range(6)]


@dataclass(frozen=True)
class Body(Box):
    """A rigid body made of ``shapes`` (body frame); without a ``mass`` it is a fixed fixture.

    ``com`` is the centre of mass in the body frame; ``friction`` is static and dynamic friction for every
    shape; ``contact_offset`` (m) is how far from a surface PhysX starts generating contacts (small, so thin
    parts in narrow gaps do not touch both walls at once); ``visual_only`` lists shape indices that are seen but
    do not collide."""

    shapes: tuple[Shape, ...] = ()
    com: tuple[float, float, float] = (0.0, 0.0, 0.0)
    friction: float = 0.8
    contact_offset: float = 0.001
    visual_only: tuple[int, ...] = ()
    solver_iterations: int = 16
    meta: dict = field(default_factory=dict, compare=False, hash=False)

    def _check(self) -> None:
        if not self.name.isidentifier():
            raise ValueError(f"object name {self.name!r} must be a Python identifier")
        if not self.shapes:
            raise ValueError(f"{self.name}: a Body needs at least one shape")
        if self.mass is not None and not self.mass > 0:
            raise ValueError(f"{self.name}.mass must be positive or None")

    def bounds_local(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.min([np.array(s.centre) - s.half_extent() for s in self.shapes], axis=0)
        hi = np.max([np.array(s.centre) + s.half_extent() for s in self.shapes], axis=0)
        return lo, hi


def body(name, pos, shapes, mass=None, yaw=0.0, com=(0.0, 0.0, 0.0), friction=0.8, contact_offset=0.001,
         visual_only=(), color=(0.5, 0.5, 0.5), solver_iterations=16) -> Body:
    """Build a :class:`Body`; ``size`` is filled in from the shapes' extent."""
    shapes = tuple(shapes)
    lo = np.min([np.array(s.centre) - s.half_extent() for s in shapes], axis=0)
    hi = np.max([np.array(s.centre) + s.half_extent() for s in shapes], axis=0)
    return Body(name=name, pos=tuple(map(float, pos)), size=tuple((hi - lo).tolist()), mass=mass, yaw=float(yaw),
                color=tuple(color), shapes=shapes, com=tuple(map(float, com)), friction=float(friction),
                contact_offset=float(contact_offset), visual_only=tuple(visual_only),
                solver_iterations=int(solver_iterations))


@dataclass(frozen=True)
class BodyState(ObjectState):
    """A body's origin pose (``pos``, ``quat_xyzw``), velocities and world-aligned bounds (``low``, ``high``)."""

    def rotation(self) -> np.ndarray:
        return quat_to_matrix(self.quat_xyzw)

    def to_world(self, local_point) -> np.ndarray:
        return self.pos + self.rotation() @ np.asarray(local_point, float)

    def to_local(self, world_point) -> np.ndarray:
        return self.rotation().T @ (np.asarray(world_point, float) - self.pos)

    def up(self) -> np.ndarray:
        """The body's z axis in the world."""
        return self.rotation()[:, 2]

    def tilt_deg(self) -> float:
        """Angle between the body's z axis and the vertical."""
        return float(np.degrees(np.arccos(np.clip(self.up()[2], -1.0, 1.0))))


def body_state(spec: Body, pos, quat_xyzw, lin_vel, ang_vel) -> BodyState:
    """A :class:`BodyState` with its bounds from the shapes' corners (exact for boxes, a box hull for cylinders)."""
    pos, quat = np.asarray(pos, float), np.asarray(quat_xyzw, float)
    r = quat_to_matrix(quat)
    lo, hi = spec.bounds_local()
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    world = corners @ r.T + pos
    return BodyState(pos=pos, quat_xyzw=quat, lin_vel=np.asarray(lin_vel, float), ang_vel=np.asarray(ang_vel, float),
                     low=world.min(axis=0), high=world.max(axis=0))


# -- the press-fit seal --------------------------------------------------------------------------------------------


@dataclass
class Seal:
    """A lid held in a tin's rim by ``points`` catches spaced evenly round a circle of ``radius`` (m).

    Lid frame and tin frame: z up the tin's axis. A catch sits at angle ``offset + 2 pi k / points`` at
    ``lid_z`` in the lid frame, and its seat at ``tin_z`` in the tin frame, so a seated lid has every catch on its
    seat. While a catch holds, it pulls the lid towards the tin along the tin's axis with ``hold`` newtons, and
    the tin towards the lid with the same force. It lets go for good once its lid point has been raised more than
    ``release`` (m) along the tin's axis from its seat, and all of them let go once the lid is tilted more than
    ``pop_tilt`` (deg) against the tin. ``held`` and ``free`` are the state; :meth:`clear` re-seats them."""

    lid: str
    tin: str
    points: int = 12
    radius: float = 0.031
    lid_z: float = 0.0
    tin_z: float = 0.0
    hold: float = 2.0
    release: float = 0.0012
    pop_tilt: float = 10.0
    offset: float = math.pi / 12
    # A section of the lid that has been pried loose rests ``proud`` above its seat (its plug sits on the rim's ring
    # instead of in it): each released catch pushes its point of the lid up towards that height with a soft,
    # one-sided, damped spring of at most ``support_max`` newtons, so the robot can still press the lid down or lift
    # it off. WHY: a real pried lid stays up; a rigid plug in a rigid ring cannot (the real one flexes past the lip).
    proud: float = 0.0
    # Once every catch has let go, the whole plug is out of the rim's channel and the lid rides on top of it: the
    # support then holds it ``proud_free`` above its seat (0: stay at ``proud``).
    proud_free: float = 0.0
    # ...and its plug, sitting in the top of the rim's channel, keeps it centred: while it is free and no higher than
    # ``proud_free`` + 2 mm, a soft damped spring of at most ``centre_max`` N pulls the lid's centre towards the tin's
    # axis (0: no pull).
    centre_k: float = 0.0
    centre_c: float = 0.5
    centre_max: float = 0.3
    # The catches and the support act only while the lid's centre is within ``engage`` of the tin's axis. WHY
    # (2026-11-07, sim trace): set down on the table 14 cm away at the seat's height, the lid switched the support and
    # centring back on, and their reaction, 14 cm from the tin's axis, tipped the tin over.
    engage: float = 0.012
    support_k: float = 150.0
    support_c: float = 1.0
    support_max: float = 0.25
    held: np.ndarray = field(init=False)
    lifted_max: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.clear()

    def clear(self) -> None:
        self.held = np.ones(self.points, dtype=bool)
        self.lifted_max = np.zeros(self.points)
        self.substeps = 0

    def angles(self) -> np.ndarray:
        return self.offset + 2 * np.pi * np.arange(self.points) / self.points

    def lid_points(self) -> np.ndarray:
        a = self.angles()
        return np.stack([self.radius * np.cos(a), self.radius * np.sin(a), np.full(self.points, self.lid_z)], axis=1)

    def seat_points(self) -> np.ndarray:
        p = self.lid_points()
        p[:, 2] = self.tin_z
        return p

    def lifts(self, lid_pos, lid_rot, tin_pos, tin_rot) -> tuple[np.ndarray, np.ndarray, float]:
        """(each catch's lid point in the world, its lift along the tin's axis (m), lid tilt against the tin (deg))."""
        world = self.lid_points() @ lid_rot.T + lid_pos
        seats = self.seat_points() @ tin_rot.T + tin_pos
        axis = tin_rot[:, 2]
        lift = (world - seats) @ axis
        tilt = float(np.degrees(np.arccos(np.clip(lid_rot[:, 2] @ axis, -1.0, 1.0))))
        return world, lift, tilt

    def update(self, lid_pos, lid_rot, tin_pos, tin_rot, lid_vel=None, tin_vel=None):
        """Release what has come loose, then return (force on lid, torque on lid about its origin, force on tin,
        torque on tin about its origin), world frame: the pull of the catches still holding, and the support of the
        released ones (``lid_vel``/``tin_vel``: (linear, angular) world velocities, for the support's damping)."""
        lid_pos, tin_pos = np.asarray(lid_pos, float), np.asarray(tin_pos, float)
        world, lift, tilt = self.lifts(lid_pos, lid_rot, tin_pos, tin_rot)
        axis = tin_rot[:, 2]
        off = lid_pos - tin_pos
        if np.linalg.norm(off - (off @ axis) * axis) > self.engage:
            self.held[:] = False
            self.lifted_max = np.maximum(self.lifted_max, np.maximum(lift, 0.0))
            z = np.zeros(3)
            return z, z, z, z
        self.lifted_max = np.maximum(self.lifted_max, lift)
        self.held &= lift <= self.release
        if tilt > self.pop_tilt:
            self.held[:] = False
        axis = tin_rot[:, 2]
        mag = np.where(self.held, -self.hold, 0.0)  # along the tin's axis, on the lid, per catch
        target = self.proud_free if (self.proud_free > 0 and not self.held.any()) else self.proud
        if target > 0:
            loose = ~self.held & (lift < target) & (lift > -0.002)
            if loose.any():
                rate = np.zeros(self.points)
                if lid_vel is not None and tin_vel is not None:
                    (lv, lw), (tv, tw) = lid_vel, tin_vel
                    v_lid = lv + np.cross(lw, world - lid_pos)
                    v_tin = tv + np.cross(tw, world - tin_pos)
                    rate = (v_lid - v_tin) @ axis
                push = np.clip(self.support_k * (target - lift) - self.support_c * rate, 0.0, self.support_max)
                mag = np.where(loose, push, mag)
        f_centre = np.zeros(3)
        if self.centre_k > 0 and self.proud_free > 0 and not self.held.any() and lift.max() < self.proud_free + 0.002:
            d = lid_pos - tin_pos
            d -= (d @ axis) * axis
            v = np.zeros(3)
            if lid_vel is not None and tin_vel is not None:
                v = lid_vel[0] - tin_vel[0]
                v -= (v @ axis) * axis
            f_centre = -self.centre_k * d - self.centre_c * v
            n = np.linalg.norm(f_centre)
            if n > self.centre_max:
                f_centre *= self.centre_max / n
        if not mag.any() and not f_centre.any():
            z = np.zeros(3)
            return z, z, z, z
        forces = mag[:, None] * axis[None, :]
        f_lid = forces.sum(axis=0) + f_centre
        t_lid = np.cross(world - lid_pos, forces).sum(axis=0)
        t_tin = np.cross(world - tin_pos, -forces).sum(axis=0) + np.cross(lid_pos - tin_pos, -f_centre)
        return f_lid, t_lid, -f_lid, t_tin

    @property
    def free(self) -> bool:
        return not self.held.any()


# -- tilted grips for yam_arena.demo waypoints --------------------------------------------------------------------


class _Grip(float):
    """A waypoint ``yaw`` that also carries a full TCP rotation matrix (see :func:`tilted`)."""

    matrix: np.ndarray

    def __new__(cls, matrix: np.ndarray, yaw: float = 0.0):
        obj = super().__new__(cls, yaw)
        obj.matrix = np.asarray(matrix, float)
        return obj


def tool_frame(heading: float, pitch: float, roll: float = 0.0) -> np.ndarray:
    """TCP rotation whose approach axis (+z) points along compass ``heading`` (rad, 0 = +x, pi/2 = +y), tipped
    ``pitch`` (rad) below the horizontal. With ``roll = 0`` the fingers close horizontally across it (along the
    TCP's x axis, as in ``yam_arena.demo.DOWN_GRIP``: ``tool_frame(0, pi/2)`` is ``DOWN_GRIP``); ``roll`` turns
    the grip about the approach axis."""
    z = np.array([math.cos(heading) * math.cos(pitch), math.sin(heading) * math.cos(pitch), -math.sin(pitch)])
    x = np.array([math.sin(heading), -math.cos(heading), 0.0])
    y = np.cross(z, x)
    c, s = math.cos(roll), math.sin(roll)
    x, y = c * x + s * y, -s * x + c * y
    return np.column_stack([x, y, z])


def tilted(heading: float, pitch: float, roll: float = 0.0) -> float:
    """A ``Waypoint.yaw`` value for a grip whose approach axis points along ``heading``, ``pitch`` below the
    horizontal (rad), turned ``roll`` about it. ``pitch = pi/2, roll = 0`` is the straight-down grip."""
    return _Grip(tool_frame(heading, pitch, roll), yaw=heading)


_orig_grip_rotation = _demo.grip_rotation


def _grip_rotation(yaw: float = 0.0) -> np.ndarray:
    m = getattr(yaw, "matrix", None)
    return m.copy() if m is not None else _orig_grip_rotation(yaw)


_demo.grip_rotation = _grip_rotation  # ArmPlan.target looks this name up at call time


# -- registries -------------------------------------------------------------------------------------------------

_SPECS: dict[str, Body] = {}
_SEALS: list[Seal] = []
_RESET: list = []


def add_seal(seal: Seal) -> Seal:
    _SEALS.append(seal)
    return seal


def on_reset(fn):
    """Register ``fn()`` to run at every episode reset, before the objects settle."""
    _RESET.append(fn)
    return fn


def register(*specs: Body) -> None:
    for s in specs:
        _SPECS[s.name] = s


# -- Isaac Sim side (lazy) ----------------------------------------------------------------------------------------


def _spawn_body(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    """Author a Body under ``prim_path``: one Xform that is the rigid body (if it has a mass), with its shapes as
    children."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material, bind_visual_material
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    from yam_arena.station import linear_rgb

    spec = _SPECS[cfg.body_name]
    stage = get_current_stage()
    root = UsdGeom.Xform.Define(stage, prim_path)
    root.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
    x, y, z, w = orientation if orientation is not None else (0.0, 0.0, 0.0, 1.0)
    root.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, x, y, z))
    prim = root.GetPrim()
    if spec.mass is not None:
        UsdPhysics.RigidBodyAPI.Apply(prim)
        mass = UsdPhysics.MassAPI.Apply(prim)
        mass.CreateMassAttr(spec.mass)
        mass.CreateCenterOfMassAttr(Gf.Vec3f(*spec.com))
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rb.CreateMaxDepenetrationVelocityAttr(0.5)
        rb.CreateSolverPositionIterationCountAttr(spec.solver_iterations)
        rb.CreateSolverVelocityIterationCountAttr(2)

    material = f"{prim_path}/Looks/physics"
    pm = sim_utils.RigidBodyMaterialCfg(static_friction=spec.friction, dynamic_friction=spec.friction)
    pm.func(material, pm)
    looks: dict[Color, str] = {}

    def look(color: Color) -> str:
        if color not in looks:
            path = f"{prim_path}/Looks/c{len(looks)}"
            vis = sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(color), roughness=0.45)
            vis.func(path, vis)
            looks[color] = path
        return looks[color]

    for i, s in enumerate(spec.shapes):
        path = f"{prim_path}/s{i}"
        if s.kind == "box":
            geom = UsdGeom.Cube.Define(stage, path)
            geom.CreateSizeAttr(1.0)
        else:
            geom = UsdGeom.Cylinder.Define(stage, path)
            geom.CreateAxisAttr(s.axis)
            geom.CreateRadiusAttr(s.size[0])
            geom.CreateHeightAttr(s.size[1])
        geom.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*s.centre))
        if s.yaw:
            geom.AddRotateZOp().Set(math.degrees(s.yaw))
        if s.kind == "box":
            geom.AddScaleOp().Set(Gf.Vec3f(*s.size))
        shape = geom.GetPrim()
        if i not in spec.visual_only:
            UsdPhysics.CollisionAPI.Apply(shape)
            cp = PhysxSchema.PhysxCollisionAPI.Apply(shape)
            cp.CreateContactOffsetAttr(spec.contact_offset)
            cp.CreateRestOffsetAttr(0.0)
            bind_physics_material(path, material, stage=stage)
        bind_visual_material(path, look(s.color), stage=stage)
    return prim


def _arena_object(spec: Body):
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    @configclass
    class BodySpawnerCfg(SpawnerCfg):
        func: object = clone(_spawn_body)
        body_name: str = ""

    return Object(
        name=spec.name,
        object_type=ObjectType.RIGID if spec.mass is not None else ObjectType.BASE,
        spawner_cfg=BodySpawnerCfg(body_name=spec.name),
        initial_pose=Pose(position_xyz=tuple(spec.pos), rotation_xyzw=yaw_quat(spec.yaw)),
    )


def _t(x):
    return getattr(x, "torch", x)


def _np(x) -> np.ndarray:
    return _t(x).detach().cpu().numpy().astype(np.float64)


def _rigid_pose(obj) -> tuple[np.ndarray, np.ndarray]:
    pose = _np(obj.data.root_link_pose_w)[0]
    return pose[:3], quat_to_matrix(pose[3:7])


def _states(environment, names) -> dict[str, BodyState]:
    base = environment.unwrapped
    origin = _np(base.scene.env_origins)[0]
    out = {}
    for n in names:
        spec = _SPECS[n]
        if spec.mass is None:
            out[n] = body_state(spec, spec.pos, yaw_quat(spec.yaw), np.zeros(3), np.zeros(3))
            continue
        obj = base.scene.rigid_objects[n]
        pose = _np(obj.data.root_link_pose_w)[0]
        vel = _np(obj.data.root_link_vel_w)[0]
        out[n] = body_state(spec, pose[:3] - origin, pose[3:7], vel[:3], vel[3:6])
    return out


_ENV: dict = {"env": None}
FINGER_LINKS = ("gripper", "tip_left", "tip_right")  # the palm and both fingers
ROBOTS = ("left_robot", "right_robot")


def finger_points() -> np.ndarray:
    """World positions (n, 3) of the vertices of both arms' gripper collision hulls (palm and fingers; the URDF's
    CoACD hulls), for a scorer to tell whether a gripper still touches a part. Empty before the first reset and
    offline."""
    environment = _ENV["env"]
    if environment is None:
        return np.zeros((0, 3))
    from yam_arena.analysis import world_hulls
    from yam_arena.runtime import link_poses

    scene = environment.unwrapped.scene
    pts = [h.vertices for name in ROBOTS for _, h in world_hulls(link_poses(scene[name]), list(FINGER_LINKS))]
    return np.vstack(pts) if pts else np.zeros((0, 3))


def _install_seal_forces(environment) -> None:
    """Wrap each sealed lid's write_data_to_sim, which Isaac Lab calls before every physics substep."""
    import warp as wp

    base = environment.unwrapped
    for seal in _SEALS:
        lid, tin = base.scene.rigid_objects[seal.lid], base.scene.rigid_objects[seal.tin]
        if getattr(lid, "_tinkit_seal", None) is seal:
            continue
        orig = lid.write_data_to_sim

        def write(seal=seal, lid=lid, tin=tin, orig=orig):
            orig()
            lp, lr = _rigid_pose(lid)
            tp, tr = _rigid_pose(tin)
            lv, tv = _np(lid.data.root_link_vel_w)[0], _np(tin.data.root_link_vel_w)[0]
            before = int(seal.held.sum())
            f_lid, t_lid, f_tin, t_tin = seal.update(lp, lr, tp, tr, (lv[:3], lv[3:6]), (tv[:3], tv[3:6]))
            seal.substeps += 1
            if DEBUG and int(seal.held.sum()) != before:
                deg = np.degrees(seal.angles()[~seal.held]).round().astype(int).tolist()
                print(f"[tinkit] substep {seal.substeps}: {int(seal.held.sum())} catches hold; free at {deg} deg; "
                      f"max lift {seal.lifted_max.max() * 1000:.2f} mm", flush=True)
            if not f_lid.any() and not t_lid.any():
                return
            dev = base.device
            for obj, f, t in ((lid, f_lid, t_lid), (tin, f_tin, t_tin)):
                # WHY: the PhysX tensor frontend here is Warp: it takes flat float32 warp arrays, as Isaac Lab's
                # own wrench composer passes (a torch tensor fails inside warp's type_ctype).
                obj.root_view.apply_forces_and_torques_at_position(
                    force_data=wp.array(f.astype(np.float32), dtype=wp.float32, device=dev),
                    torque_data=wp.array(t.astype(np.float32), dtype=wp.float32, device=dev),
                    position_data=None,
                    indices=obj._ALL_INDICES,
                    is_global=True,
                )

        lid.write_data_to_sim = write
        lid._tinkit_seal = seal


def _patch_scene(scene) -> None:
    if getattr(scene, "_tinkit_patched", False):
        return
    orig_task_object, orig_states = scene.task_object, scene.object_states

    def task_object(spec):
        if isinstance(spec, Body):
            _SPECS[spec.name] = spec
            return _arena_object(spec)
        return orig_task_object(spec)

    def object_states(environment, names):
        plain = [n for n in names if n not in _SPECS]
        out = orig_states(environment, plain) if plain else {}
        out.update(_states(environment, [n for n in names if n in _SPECS]))
        return {n: out[n] for n in names}

    scene.task_object = task_object
    scene.object_states = object_states
    scene._tinkit_patched = True


def _patch_embodiment(ie) -> None:
    cls = ie.YamArenaEmbodiment
    if getattr(cls, "_tinkit_patched", False):
        return
    base_reset = cls.reset

    def reset(self, *a, **k):
        # WHY: the seal and any latched violations must be re-armed before the env reset re-seats the lid and the
        # settle steps run, or a lid released in the last trial would start this one loose.
        for seal in _SEALS:
            seal.clear()
        for fn in _RESET:
            fn()
        if getattr(self, "_env", None) is not None:
            _install_seal_forces(self._env)
        obs = base_reset(self, *a, **k)
        _install_seal_forces(self._env)  # the first reset builds the env
        _ENV["env"] = self._env
        for seal in _SEALS:  # the settle steps may have nudged the lid: re-arm on the settled pose
            seal.clear()
        for fn in _RESET:
            fn()
        return obs

    cls.reset = reset
    cls._tinkit_patched = True


class _Finder(importlib.abc.MetaPathFinder):
    targets = {"yam_arena.scene": _patch_scene, "yam_arena.inspect_embodiment": _patch_embodiment}

    def find_spec(self, fullname, path, target=None):
        if fullname not in self.targets:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        loader, patch = spec.loader, self.targets[fullname]
        orig_exec = loader.exec_module

        def exec_module(module):
            orig_exec(module)
            patch(module)

        loader.exec_module = exec_module
        return spec


def install() -> None:
    """Hook yam_arena: now for modules already imported, otherwise when they are. Idempotent."""
    for name, patch in _Finder.targets.items():
        if name in sys.modules:
            patch(sys.modules[name])
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
