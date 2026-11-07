"""glazekit: compound rigid bodies, hinged levers and hand suction lifters for a yam_arena task file.

Written for the glass-pane task in this folder, from the helper of our earlier paint-tin task. Importing it is
cheap (NumPy only): everything that needs Isaac Sim is imported lazily, inside hooks that run once the simulator
has started.

What it adds to a plain task file:

- :class:`Body`: one rigid body (or a fixed fixture, without a mass) made of several boxes and cylinders, each
  placed and turned freely in the body's frame. It subclasses ``task_file.Box`` only so that the task-file
  loader accepts it in ``OBJECTS``; ``size`` is its bounding box and ``pos`` is the body's origin.
- :class:`Hinge`: a PhysX revolute joint between two bodies (a lifter and its lever).
- :class:`Detent`: an over-centre spring on a hinge, so a lever snaps shut or open and stays there.
- :class:`Suction`: a suction lifter's cup on a flat part: it seals when its lever is turned down while the cup
  lies flat and fully on the part, then holds it with stiff springs until the lever is lifted or it is pulled off.
- Force models run before every physics substep (240 Hz), so a quick motion cannot skip them.
- :func:`tilted`: a waypoint orientation for ``yam_arena.demo`` that is not a straight-down grip.
- :func:`finger_points`: the grippers' collision-hull vertices, for scoring whether a gripper touches a part.
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

__all__ = ["Shape", "box", "cyl", "ring", "prism12", "Body", "body", "BodyState", "Hinge", "Detent", "Suction", "add_hinge", "add_model", "tilted",
           "on_reset", "finger_points", "install"]

Color = tuple[float, float, float]
DEBUG = bool(__import__("os").environ.get("GLAZEKIT_DEBUG"))  # print prim paths and joint authoring


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
    quat: tuple[float, float, float, float] | None = None  # xyzw start orientation; overrides ``yaw`` if given
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
         visual_only=(), color=(0.5, 0.5, 0.5), solver_iterations=16, quat=None) -> Body:
    """Build a :class:`Body`; ``size`` is filled in from the shapes' extent."""
    shapes = tuple(shapes)
    lo = np.min([np.array(s.centre) - s.half_extent() for s in shapes], axis=0)
    hi = np.max([np.array(s.centre) + s.half_extent() for s in shapes], axis=0)
    return Body(name=name, pos=tuple(map(float, pos)), size=tuple((hi - lo).tolist()), mass=mass, yaw=float(yaw),
                color=tuple(color), shapes=shapes, com=tuple(map(float, com)), friction=float(friction),
                contact_offset=float(contact_offset), visual_only=tuple(visual_only),
                solver_iterations=int(solver_iterations), quat=None if quat is None else tuple(map(float, quat)))


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


# -- hinges and per-substep force models -----------------------------------------------------------------------


@dataclass(frozen=True)
class Hinge:
    """A revolute joint between two bodies, authored as a PhysX joint when the child is spawned.

    Both bodies' frames coincide when the joint reads 0. ``pivot`` and ``axis`` are in that shared frame (m, unit
    vector). The child is spawned already turned ``start`` (deg) about the axis, and can turn from ``lower`` to
    ``upper`` (deg). The two bodies do not collide with each other."""

    parent: str
    child: str
    pivot: tuple[float, float, float]
    axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    start: float = 0.0
    lower: float = 0.0
    upper: float = 90.0

    def rotation(self, deg: float) -> np.ndarray:
        """Rotation about the axis by ``deg``."""
        a = np.asarray(self.axis, float)
        a /= np.linalg.norm(a)
        t = math.radians(deg)
        k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
        return np.eye(3) + math.sin(t) * k + (1 - math.cos(t)) * (k @ k)

    def child_pose(self, parent_pos, parent_yaw: float) -> tuple[tuple, tuple]:
        """World (position, xyzw quaternion) of the child at ``start``, its parent at ``parent_pos``/``parent_yaw``."""
        rp = quat_to_matrix(yaw_quat(parent_yaw))
        r = self.rotation(self.start)
        piv = np.asarray(self.pivot, float)
        pos = np.asarray(parent_pos, float) + rp @ (piv - r @ piv)
        return tuple(pos.tolist()), matrix_to_quat(rp @ r)

    def angle(self, parent_rot: np.ndarray, child_rot: np.ndarray) -> float:
        """The joint's reading (deg) from the two bodies' world rotations."""
        rel = parent_rot.T @ child_rot
        a = np.asarray(self.axis, float)
        a /= np.linalg.norm(a)
        u = np.cross(a, [0.0, 0.0, 1.0] if abs(a[2]) < 0.9 else [1.0, 0.0, 0.0])
        u /= np.linalg.norm(u)
        v = rel @ u
        return math.degrees(math.atan2(np.cross(u, v) @ a, u @ v))


def matrix_to_quat(r: np.ndarray) -> tuple[float, float, float, float]:
    """xyzw quaternion of a rotation matrix."""
    w = math.sqrt(max(0.0, 1 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + r[0, 0] - r[1, 1] - r[2, 2])) / 2, r[2, 1] - r[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - r[0, 0] + r[1, 1] - r[2, 2])) / 2, r[0, 2] - r[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - r[0, 0] - r[1, 1] + r[2, 2])) / 2, r[1, 0] - r[0, 1])
    return (x, y, z, w)


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


@dataclass
class Detent:
    """An over-centre spring on a hinge: below ``mid`` (deg) it turns the child towards ``low``, above it towards
    ``high``, with ``k`` N m/rad and ``c`` N m s/rad, at most ``max_torque`` N m. A real lever's cam does this: it
    snaps shut or open and stays there."""

    hinge: Hinge
    low: float = 0.0
    high: float = 60.0
    mid: float = 25.0
    k: float = 0.08
    c: float = 0.002
    max_torque: float = 0.04

    @property
    def bodies(self) -> tuple[str, str]:
        return (self.hinge.parent, self.hinge.child)

    def clear(self) -> None:
        pass

    def update(self, fr: dict, acc: dict) -> None:
        h = self.hinge
        p, c = fr[h.parent], fr[h.child]
        ang = h.angle(p.rot, c.rot)
        axis = p.rot @ (np.asarray(h.axis, float) / np.linalg.norm(h.axis))
        target = self.low if ang < self.mid else self.high
        rate = (c.spin - p.spin) @ axis
        tau = float(np.clip(self.k * math.radians(target - ang) - self.c * rate, -self.max_torque, self.max_torque))
        for name, sign in ((h.child, 1.0), (h.parent, -1.0)):
            _, t = acc.setdefault(name, [np.zeros(3), np.zeros(3)])
            t += sign * tau * axis


@dataclass
class Suction:
    """A hand suction lifter's cup on a flat part.

    Lifter frame: origin at the centre of the cup's face, z out of its back (up when the cup lies face down). Part
    frame: z up through its top face, which is at ``part_top``; the face is ``part_half`` (half x, half y) about the
    origin. The cup seals when its lever (``hinge``, the lifter its parent) is turned down below ``locked_below`` (deg)
    while the cup lies on the part's top face: tilted under ``flat_deg`` against it, its face within ``gap_max`` above
    the face (or up to 1 mm into it), and its rim at least ``margin`` inside every edge. Turned down anywhere else, it
    does not seal, and the lever must be lifted again before it can. While sealed, three stiff springs at ``radius_k``
    on the cup's face tie it to the points of the part it sealed on (``k`` N/m, ``c`` N s/m). If any spring has to pull
    or push harder than ``max_force`` N, the cup pops off. Lifting the lever above ``locked_below`` releases it.
    ``sealed`` is the state; ``ever_sealed`` latches."""

    lifter: str
    hinge: Hinge
    part: str
    part_half: tuple[float, float]
    part_top: float
    radius: float = 0.0575
    radius_k: float = 0.040
    locked_below: float = 10.0
    flat_deg: float = 3.0
    gap_max: float = 0.003
    margin: float = 0.005
    k: float = 4000.0
    c: float = 20.0
    max_force: float = 40.0
    sealed: bool = field(init=False, default=False)
    armed: bool = field(init=False, default=True)
    ever_sealed: bool = field(init=False, default=False)
    popped: int = field(init=False, default=0)
    anchors: np.ndarray = field(init=False, default=None)

    @property
    def bodies(self) -> tuple[str, str, str]:
        return (self.lifter, self.hinge.child, self.part)

    def clear(self) -> None:
        self.sealed, self.armed, self.ever_sealed, self.popped, self.anchors = False, True, False, 0, None

    def rim(self) -> np.ndarray:
        a = np.radians([90.0, 210.0, 330.0])
        return np.stack([self.radius_k * np.cos(a), self.radius_k * np.sin(a), np.zeros(3)], axis=1)

    def seats(self, lifter: Frame, part: Frame) -> bool:
        """Whether the cup lies flat, fully on the part's top face."""
        tilt = math.degrees(math.acos(float(np.clip(lifter.rot[:, 2] @ part.rot[:, 2], -1.0, 1.0))))
        o = part.local(lifter.pos)
        hx, hy = self.part_half
        inside = abs(o[0]) <= hx - self.radius - self.margin and abs(o[1]) <= hy - self.radius - self.margin
        return bool(tilt < self.flat_deg and -0.001 <= o[2] - self.part_top <= self.gap_max and inside)

    def update(self, fr: dict, acc: dict) -> None:
        lifter, lever, part = fr[self.lifter], fr[self.hinge.child], fr[self.part]
        ang = self.hinge.angle(lifter.rot, lever.rot)
        if ang >= self.locked_below:
            self.sealed, self.armed = False, True
            return
        if self.armed:
            self.armed = False
            if self.seats(lifter, part):
                self.sealed = self.ever_sealed = True
                self.anchors = np.array([part.local(lifter.world(p)) for p in self.rim()])
        if not self.sealed:
            return
        forces = []
        for p, q in zip(self.rim(), self.anchors):
            a, b = lifter.world(p), part.world(q)
            f = self.k * (b - a) + self.c * (part.point_vel(b) - lifter.point_vel(a))
            if np.linalg.norm(f) > self.max_force:
                self.sealed = False
                self.popped += 1
                return
            forces.append((a, b, f))
        for a, b, f in forces:
            _wrench(acc, self.lifter, lifter, f, a)
            _wrench(acc, self.part, part, -f, b)


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
_HINGES: dict[str, Hinge] = {}  # by child
_MODELS: list = []
_RESET: list = []


def add_hinge(hinge: Hinge) -> Hinge:
    _HINGES[hinge.child] = hinge
    return hinge


def add_model(model):
    """Register a force model: ``bodies`` (names), ``clear()`` and ``update(frames, acc)``, run every substep."""
    _MODELS.append(model)
    return model


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
    hinge = _HINGES.get(cfg.body_name)
    if hinge is not None:
        _author_hinge(stage, prim_path, hinge)
    return prim


def _author_hinge(stage, prim_path: str, h: Hinge) -> None:
    """A PhysX revolute joint from the parent (a sibling prim) to this child. The joint frame's x axis is the
    hinge axis, at the pivot, the same in both bodies' frames (they coincide when the joint reads 0)."""
    from pxr import Gf, UsdPhysics

    parent_path = prim_path.rsplit("/", 1)[0] + "/" + h.parent
    a = np.asarray(h.axis, float)
    a /= np.linalg.norm(a)
    x = np.array([1.0, 0.0, 0.0])
    v = np.cross(x, a)
    if np.linalg.norm(v) < 1e-9:
        q = (0.0, 0.0, 0.0, 1.0) if a[0] > 0 else (0.0, 0.0, 1.0, 0.0)
    else:
        ang = math.acos(float(np.clip(x @ a, -1.0, 1.0)))
        v = v / np.linalg.norm(v) * math.sin(ang / 2)
        q = (v[0], v[1], v[2], math.cos(ang / 2))
    rot = Gf.Quatf(float(q[3]), float(q[0]), float(q[1]), float(q[2]))
    j = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/hinge")
    j.CreateBody0Rel().SetTargets([parent_path])
    j.CreateBody1Rel().SetTargets([prim_path])
    j.CreateAxisAttr("X")
    j.CreateLocalPos0Attr(Gf.Vec3f(*map(float, h.pivot)))
    j.CreateLocalRot0Attr(rot)
    j.CreateLocalPos1Attr(Gf.Vec3f(*map(float, h.pivot)))
    j.CreateLocalRot1Attr(rot)
    j.CreateLowerLimitAttr(float(h.lower))
    j.CreateUpperLimitAttr(float(h.upper))
    j.CreateCollisionEnabledAttr(False)
    if DEBUG:
        print(f"[glazekit] hinge {parent_path} -> {prim_path}", flush=True)


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
    if getattr(anchor, "_glazekit_hooked", False):
        return
    orig = anchor.write_data_to_sim

    def write():
        orig()
        frames = {n: _frame(o, _SPECS[n]) for n, o in objs.items()}
        acc: dict = {}
        for m in _MODELS:
            m.update(frames, acc)
        dev = base.device
        for n, (f, t) in acc.items():
            if not f.any() and not t.any():
                continue
            o = objs[n]
            # WHY warp arrays: the PhysX tensor frontend here takes flat float32 warp arrays (a torch tensor fails
            # inside warp's type_ctype); see the paint-tin task's tinkit.
            o.root_view.apply_forces_and_torques_at_position(
                force_data=wp.array(f.astype(np.float32), dtype=wp.float32, device=dev),
                torque_data=wp.array(t.astype(np.float32), dtype=wp.float32, device=dev),
                position_data=None,
                indices=o._ALL_INDICES,
                is_global=True,
            )

    anchor.write_data_to_sim = write
    anchor._glazekit_hooked = True


def _patch_scene(scene) -> None:
    if getattr(scene, "_glazekit_patched", False):
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
    scene._glazekit_patched = True


def _patch_embodiment(ie) -> None:
    cls = ie.YamArenaEmbodiment
    if getattr(cls, "_glazekit_patched", False):
        return
    base_reset = cls.reset

    def reset(self, *a, **k):
        # WHY: the force models' state and any latched violations must be cleared before the env reset and its
        # settle steps run, or a cup sealed in the last trial would start this one stuck to the glass.
        for m in _MODELS:
            m.clear()
        for fn in _RESET:
            fn()
        first = getattr(self, "_env", None) is None
        if not first:
            _install_forces(self._env)
        obs = base_reset(self, *a, **k)
        _install_forces(self._env)  # the first reset builds the env
        if first and _MODELS:
            # WHY reset twice the first time (2026-11-07, sim trace): the first reset builds the env and runs its settle
            # steps before the force hooks exist, so the levers' detents were off and both levers fell shut.
            for m in _MODELS:
                m.clear()
            obs = base_reset(self, *a, **k)
        _ENV["env"] = self._env
        for m in _MODELS:  # whatever the settle steps did, every trial starts unsealed
            m.clear()
        for fn in _RESET:
            fn()
        return obs

    cls.reset = reset
    cls._glazekit_patched = True


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
