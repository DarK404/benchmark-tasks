"""filekit: compound rigid bodies and a file cutting a block, for a yam_arena task file.

Written for the filing task in this folder, from the helpers of our earlier tasks. Importing it is cheap
(NumPy only): everything that needs Isaac Sim is imported lazily, inside hooks that run once the simulator has
started.

What it adds to a plain task file:

- :class:`Body`: one rigid body (or a fixed fixture, without a mass) made of several boxes and cylinders; a colour
  with a fourth (alpha) component below 1 is drawn see-through (glass).
- :class:`Filing`: a flat file cutting the top of a block: the file stands on springs we compute, and each section
  of the top is lowered as it is filed; bodies without gravity (:func:`no_gravity`) are moved to draw the top.
- Force models run before every physics substep (240 Hz), so a quick motion cannot skip them.
- :func:`finger_points`: the grippers' collision-hull vertices, for scoring whether a gripper touches a part.
- :func:`on_reset`: functions to run at every episode reset, e.g. to clear latched scoring state.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import math
import sys
from dataclasses import dataclass, field

import numpy as np

from yam_arena.task_file import Box, ObjectState

__all__ = ["Shape", "box", "cyl", "ring", "prism12", "Body", "body", "BodyState", "Filing", "no_gravity",
           "add_model", "on_reset", "register", "finger_points", "install"]

Color = tuple[float, ...]  # RGB, or RGBA for a see-through (glass) look


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
    return [box((r_mid * math.cos(a), r_mid * math.sin(a), (z0 + z1) / 2), (chord, width, h), color,
                yaw=a + math.pi / 2)
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
    quat: tuple[float, float, float, float] | None = None  # xyzw start orientation; overrides ``yaw`` if given
    inertia: tuple[float, float, float] | None = None  # principal moments (kg m2) about body-aligned axes at the com
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
         visual_only=(), color=(0.5, 0.5, 0.5), solver_iterations=16, quat=None, inertia=None) -> Body:
    """Build a :class:`Body`; ``size`` is filled in from the shapes' extent."""
    shapes = tuple(shapes)
    lo = np.min([np.array(s.centre) - s.half_extent() for s in shapes], axis=0)
    hi = np.max([np.array(s.centre) + s.half_extent() for s in shapes], axis=0)
    return Body(name=name, pos=tuple(map(float, pos)), size=tuple((hi - lo).tolist()), mass=mass, yaw=float(yaw),
                color=tuple(color), shapes=shapes, com=tuple(map(float, com)), friction=float(friction),
                contact_offset=float(contact_offset), visual_only=tuple(visual_only),
                solver_iterations=int(solver_iterations), quat=None if quat is None else tuple(map(float, quat)),
                inertia=None if inertia is None else tuple(map(float, inertia)))


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


# -- per-substep force models -----------------------------------------------------------------------------------


@dataclass
class Frame:
    """A body during a physics substep: origin, rotation, origin velocity, angular velocity, centre of mass."""

    pos: np.ndarray
    rot: np.ndarray
    vel: np.ndarray
    spin: np.ndarray
    com: np.ndarray

    def world(self, local) -> np.ndarray:
        return self.pos + self.rot @ np.asarray(local, float)

    def local(self, world) -> np.ndarray:
        return self.rot.T @ (np.asarray(world, float) - self.pos)

    def point_vel(self, world_point) -> np.ndarray:
        return self.vel + np.cross(self.spin, np.asarray(world_point, float) - self.pos)


def _wrench(acc: dict, name: str, frame: Frame, force, at) -> None:
    """Add ``force`` (N, world) acting at world point ``at`` to body ``name``: as a force and a torque about its
    centre of mass, which is how PhysX applies them."""
    f, t = acc.setdefault(name, [np.zeros(3), np.zeros(3)])
    f += force
    t += np.cross(np.asarray(at, float) - frame.com, force)


class Filing:
    """A flat file cutting the top of a block, which is split into sections along the stroke.

    The file (body ``tool``) cuts with the flat bottom of its blade: in its own frame the plane z = 0, from
    ``blade_y[0]`` to ``blade_y[1]`` along its length (+y towards the tip) and +-``half_width`` across. Each section
    of the block's top has a footprint (world x, y ranges, task frame) and a height ``h`` (task frame), starting at
    ``h0``. Every substep, where the blade's bottom lies below a section's top over the section's centre, the block
    pushes the file up there with a spring (``k`` N/m, ``damping`` N s/m) and resists its sliding with friction
    ``mu``; the file stands on these springs, not on the simulator's contacts, so the cut is never a thin contact.
    While the file slides towards its tip (a cutting stroke), the section loses ``wear`` m of height per newton
    pressed and metre slid (Archard's wear law); sliding back cuts nothing. ``surfaces`` (bodies without gravity or
    collisions) are moved so each one's top shows its section's height."""

    def __init__(self, tool, blade_y, half_width, sections, h0, k=2000.0, damping=6.0, mu=0.5, wear=6e-4,
                 surfaces=None, floor=None):
        self.tool, self.blade_y, self.half_width = tool, tuple(blade_y), float(half_width)
        self.sections = [tuple(map(float, (x0, x1, y0, y1))) for x0, x1, y0, y1 in sections]
        self.h0, self.k, self.damping, self.mu, self.wear = float(h0), float(k), float(damping), float(mu), float(wear)
        self.surfaces = list(surfaces or [])
        self.floor = float(floor) if floor is not None else -math.inf
        self.surface_depth = 0.05  # each surface body's box reaches this far below its top
        self.origin = np.zeros(3)
        self.dt = 1.0 / 240.0
        self.poses: dict = {}
        self.clear()

    @property
    def bodies(self) -> list[str]:
        return [self.tool] + self.surfaces

    def clear(self) -> None:
        self.h = np.full(len(self.sections), self.h0)
        self.force = np.zeros(len(self.sections))
        self.cut = 0.0  # metres slid while cutting, all sections together

    def _blade_z(self, fr: Frame, x: float, y: float):
        """The blade bottom's world height over world (x, y), or None if (x, y) is not under the blade."""
        n = fr.rot[:, 2]
        if n[2] < 0.5:
            return None
        z = fr.pos[2] - (n[0] * (x - fr.pos[0]) + n[1] * (y - fr.pos[1])) / n[2]
        local = fr.local((x, y, z))
        if abs(local[0]) > self.half_width or not self.blade_y[0] <= local[1] <= self.blade_y[1]:
            return None
        return z

    def update(self, frames: dict, acc: dict) -> None:
        fr = frames.get(self.tool)
        if fr is None:
            return
        o = self.origin
        axis = fr.rot[:, 1]
        self.force[:] = 0.0
        for i, (x0, x1, y0, y1) in enumerate(self.sections):
            cy = (y0 + y1) / 2 + o[1]
            top = self.h[i] + o[2]
            # three contact points across the section (its edges and middle): a file a little off the middle still
            # bears on the part of the section under it; each point carries a third of the section's stiffness
            for cx in (x0 + 0.15 * (x1 - x0) + o[0], (x0 + x1) / 2 + o[0], x1 - 0.15 * (x1 - x0) + o[0]):
                zb = self._blade_z(fr, cx, cy)
                if zb is None:
                    continue
                d = top - zb
                if d <= 0.0:
                    continue
                at = np.array([cx, cy, zb])
                v = fr.point_vel(at)
                n_force = max(0.0, (self.k * d - self.damping * v[2]) / 3.0)
                vt = np.array([v[0], v[1], 0.0])
                speed = float(np.linalg.norm(vt))
                fric = -self.mu * n_force * vt / max(speed, 0.005)
                _wrench(acc, self.tool, fr, np.array([0.0, 0.0, n_force]) + fric, at)
                self.force[i] += n_force
                along = float(np.dot(vt, axis))
                if along > 0.0:  # pushed towards the tip: a cutting stroke
                    slid = along * self.dt
                    self.h[i] = max(self.floor, self.h[i] - self.wear * n_force * slid)
                    self.cut += slid / 3.0
        self.poses = {s: (((x0 + x1) / 2 + o[0], (y0 + y1) / 2 + o[1], self.h[i] + o[2]), (0.0, 0.0, 0.0, 1.0))
                      for i, (s, (x0, x1, y0, y1)) in enumerate(zip(self.surfaces, self.sections))}


def no_gravity(name: str) -> None:
    _NO_GRAVITY.add(name)


_SPECS: dict[str, Body] = {}
_MODELS: list = []
METALLIC: dict = {}  # colour (RGB) -> roughness: shapes of these colours are drawn as bare metal
_NO_GRAVITY: set = set()  # bodies that ignore gravity (moved by a model, e.g. the filed top's sections)
_RESET: list = []


def add_model(model):
    """Register a force model: ``bodies`` (names), ``clear()`` and ``update(frames, acc)``, run every substep."""
    _MODELS.append(model)
    return model


def on_reset(fn):
    """Register ``fn()`` to run at every episode reset, before the objects settle."""
    _RESET.append(fn)
    return fn


def register(*specs) -> None:
    for s in specs:
        _SPECS[s.name] = s


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
        if spec.inertia is not None:
            mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*spec.inertia))
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        rb.CreateMaxDepenetrationVelocityAttr(0.5)
        rb.CreateSolverPositionIterationCountAttr(spec.solver_iterations)
        rb.CreateSolverVelocityIterationCountAttr(2)
        if spec.name in _NO_GRAVITY:
            rb.CreateDisableGravityAttr(True)

    material = f"{prim_path}/Looks/physics"
    pm = sim_utils.RigidBodyMaterialCfg(static_friction=spec.friction, dynamic_friction=spec.friction)
    pm.func(material, pm)
    looks: dict[Color, str] = {}

    def look(color: Color) -> str:
        if color not in looks:
            path = f"{prim_path}/Looks/c{len(looks)}"
            rgb, alpha = color[:3], (color[3] if len(color) > 3 else 1.0)
            if alpha < 1.0:
                # WHY glass (2026-11-09, render check): RTX real-time ignores a preview surface's opacity, so a
                # see-through part (acrylic, water) is drawn with the OmniGlass material instead; its fourth colour
                # component picks the refractive index: 1.49 acrylic below 0.4, 1.33 water above.
                vis = sim_utils.GlassMdlCfg(glass_color=tuple(rgb), frosting_roughness=0.0, thin_walled=alpha < 0.4,
                                            glass_ior=1.49 if alpha < 0.4 else 1.33)
            elif tuple(rgb) in METALLIC:
                vis = sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(rgb), roughness=METALLIC[tuple(rgb)],
                                                  metallic=0.85)
            else:
                vis = sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(rgb), roughness=0.45)
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
        initial_pose=Pose(position_xyz=tuple(spec.pos), rotation_xyzw=spec.quat or yaw_quat(spec.yaw)),
    )


def _t(x):
    return getattr(x, "torch", x)


def _np(x) -> np.ndarray:
    x = _t(x)
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x.numpy(), dtype=np.float64)  # a warp array


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


def _frame(obj, spec) -> Frame:
    pose = _np(obj.data.root_link_pose_w)[0]
    vel = _np(obj.data.root_link_vel_w)[0]
    rot = quat_to_matrix(pose[3:7])
    return Frame(pos=pose[:3], rot=rot, vel=vel[:3], spin=vel[3:6], com=pose[:3] + rot @ np.asarray(spec.com, float))


def _install_forces(environment) -> None:
    """Run every force model before each physics substep (Isaac Lab calls ``write_data_to_sim`` then), from one
    hook on the first body any model names, and apply the summed wrenches to each body."""
    import warp as wp

    if not _MODELS:
        return
    base = environment.unwrapped
    names = sorted({n for m in _MODELS for n in m.bodies})
    objs = {n: base.scene.rigid_objects[n] for n in names}
    anchor = objs[names[0]]
    for m in _MODELS:
        if hasattr(m, "dt"):
            m.dt = float(base.physics_dt)
        if hasattr(m, "origin"):
            m.origin = _np(base.scene.env_origins)[0][:3]
    if getattr(anchor, "_filekit_hooked", False):
        return
    orig = anchor.write_data_to_sim

    def write():
        orig()
        frames = {n: _frame(o, _SPECS[n]) for n, o in objs.items()}
        acc: dict = {}
        for m in _MODELS:
            m.update(frames, acc)
        dev = base.device
        import torch
        for m in _MODELS:
            for n, (p, q) in getattr(m, "poses", {}).items():
                pose = torch.tensor([[*p, *q]], dtype=torch.float32, device=dev)
                objs[n].write_root_link_pose_to_sim_index(root_pose=pose)
                objs[n].write_root_link_velocity_to_sim_index(root_velocity=torch.zeros((1, 6), device=dev))
        for n, (f, t) in acc.items():
            if not f.any() and not t.any():
                continue
            o = objs[n]
            # WHY warp arrays: the PhysX tensor frontend here takes flat float32 warp arrays (a torch tensor fails
            # inside warp's type_ctype); see our paint-tin task's helper.
            o.root_view.apply_forces_and_torques_at_position(
                force_data=wp.array(f.astype(np.float32), dtype=wp.float32, device=dev),
                torque_data=wp.array(t.astype(np.float32), dtype=wp.float32, device=dev),
                position_data=None,
                indices=o._ALL_INDICES,
                is_global=True,
            )

    anchor.write_data_to_sim = write
    anchor._filekit_hooked = True


def _patch_scene(scene) -> None:
    if getattr(scene, "_filekit_patched", False):
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
    scene._filekit_patched = True


def _patch_embodiment(ie) -> None:
    cls = ie.YamArenaEmbodiment
    if getattr(cls, "_filekit_patched", False):
        return
    base_reset = cls.reset

    def clean() -> None:
        for m in _MODELS:
            m.clear()
        for fn in _RESET:
            fn()

    def reset(self, *a, **k):
        # WHY: the models' state and any latched scoring state must be cleared before the env reset and its settle
        # steps run.
        clean()
        first = getattr(self, "_env", None) is None
        if not first:
            _install_forces(self._env)
        obs = base_reset(self, *a, **k)
        _install_forces(self._env)  # the first reset builds the env
        if first and _MODELS:
            # WHY reset twice the first time (2026-11-07, our glass-pane task's sim trace): the first reset builds the
            # env and runs its settle steps before the force hooks exist, so the models miss them.
            clean()
            obs = base_reset(self, *a, **k)
        _ENV["env"] = self._env
        clean()  # whatever the settle steps did, every trial starts clean
        return obs

    cls.reset = reset
    cls._filekit_patched = True


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
