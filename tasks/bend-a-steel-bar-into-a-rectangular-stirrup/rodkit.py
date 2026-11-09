"""rodkit: compound rigid bodies and a plastically bendable rod for a yam_arena task file.

Written for the stirrup-bending task in this folder, from the helpers of our earlier tasks. Importing it is cheap
(NumPy only): everything that needs Isaac Sim is imported lazily, inside hooks that run once the simulator has
started.

What it adds to a plain task file:

- :class:`Body`: one rigid body (or a fixed fixture, without a mass) made of several boxes and cylinders.
- :class:`Rod`: a thin straight rod built as one articulation: a chain of short links joined by revolute joints
  about the rod's local z. Each joint's drive is the rod's elastic bending stiffness; once a joint's elastic moment
  passes the yield moment, its rest angle follows the bend, so the rod stays bent and springs back by its elastic
  part (perfectly plastic, a moment-rotation model of a bar bent past yield).
- Force models run before every physics substep (240 Hz), so a quick motion cannot skip them.
- :func:`tilted`: a waypoint orientation for ``yam_arena.demo`` that is not a straight-down grip.
- :func:`finger_points`: the grippers' collision-hull vertices, for scoring whether a gripper touches a part.
- :func:`on_reset`: functions to run at every episode reset, e.g. to clear latched violations.
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

__all__ = ["Shape", "box", "cyl", "ring", "prism12", "Body", "body", "BodyState", "Hinge", "add_hinge", "Rod", "rod",
           "RodState", "Bender", "add_bender", "add_model", "tilted", "on_reset", "finger_points", "install"]

Color = tuple[float, float, float]
DEBUG = bool(__import__("os").environ.get("RODKIT_DEBUG"))  # print the thread model's state now and then


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


def matrix_to_quat(r: np.ndarray) -> tuple[float, float, float, float]:
    """xyzw quaternion of a rotation matrix."""
    w = math.sqrt(max(0.0, 1 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    x = math.copysign(math.sqrt(max(0.0, 1 + r[0, 0] - r[1, 1] - r[2, 2])) / 2, r[2, 1] - r[1, 2])
    y = math.copysign(math.sqrt(max(0.0, 1 - r[0, 0] + r[1, 1] - r[2, 2])) / 2, r[0, 2] - r[2, 0])
    z = math.copysign(math.sqrt(max(0.0, 1 - r[0, 0] - r[1, 1] + r[2, 2])) / 2, r[1, 0] - r[0, 1])
    return (x, y, z, w)


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


@dataclass(frozen=True)
class Rod(Box):
    """A thin straight rod, ``links`` square-section links of ``link_len`` (m) along its local +y from its origin, the
    first link's start; ``thick`` (m) is its section. A revolute joint about the rod's local z joins each pair of links,
    at the boundary between them. ``stiffness`` (N m/rad) and ``damping`` (N m s/rad) per joint are the elastic
    bending stiffness of one link's length of rod (E I / link_len); ``yield_moment`` (N m) is the bending moment at
    which it yields. ``marks``: distances (m) from the origin where a chalk mark is drawn (seen only)."""

    links: int = 30
    link_len: float = 0.010
    thick: float = 0.003
    stiffness: float = 80.0
    damping: float = 0.05
    yield_moment: float = 0.8
    friction: float = 0.4
    contact_offset: float = 0.001
    solver_iterations: int = 32
    marks: tuple[float, ...] = ()
    link_inertia: float | None = None  # each link's principal moments (kg m2); None: from its shape and mass
    mark_color: tuple[float, float, float] = (0.95, 0.95, 0.92)

    def _check(self) -> None:
        if not self.name.isidentifier():
            raise ValueError(f"object name {self.name!r} must be a Python identifier")
        if self.links < 2 or self.mass is None or not self.mass > 0:
            raise ValueError(f"{self.name}: a Rod needs at least two links and a mass")

    def bounds_local(self) -> tuple[np.ndarray, np.ndarray]:
        h = self.thick / 2
        return np.array([-h, 0.0, -h]), np.array([h, self.links * self.link_len, h])

    @property
    def length(self) -> float:
        return self.links * self.link_len


def rod(name, pos, yaw=0.0, links=30, link_len=0.010, thick=0.003, mass=0.017, color=(0.36, 0.36, 0.38), **kw) -> Rod:
    return Rod(name=name, pos=tuple(map(float, pos)), size=(thick, links * link_len, thick), mass=float(mass),
               yaw=float(yaw), color=tuple(color), links=int(links), link_len=float(link_len), thick=float(thick),
               **kw)


@dataclass(frozen=True)
class RodState(ObjectState):
    """A rod after a control step: ``points`` (links + 1, 3) world positions of the rod's centreline at its ends and
    joints, ``angles`` (links - 1) each joint's bend (rad, positive anticlockwise about the rod's local z), ``speeds``
    each link's speed (m/s). ``pos``/``quat_xyzw`` are the first link's."""

    points: np.ndarray = None
    angles: np.ndarray = None
    speeds: np.ndarray = None

    def resting(self, max_speed: float = 0.01, max_spin: float = 0.5) -> bool:
        return bool(self.speeds is not None and self.speeds.max() < max_speed)


@dataclass
class Bender:
    """A hand bending table's lever forming a bend in a Rod round a fixed pin.

    The lever (a hinged rigid body, ``lever``) turns about the pin's vertical axis through world ``pin`` (x, y). Turned
    anticlockwise from its rest, it bends the rod round the pin: the rod wraps the pin at centreline radius ``wrap_r``,
    so the joints at the pin, starting with the one at the rod's tangent point and moving towards the free end, follow
    the lever one after another (each by up to link_len / wrap_r rad). While the lever is turned further than it has
    been (forming), it is resisted by ``moment`` N m, the bar's plastic bending moment; turned back, it is free. Let
    off, the bend relaxes by ``springback`` (deg). The bend goes the way the rod's joint angles are negative for an
    anticlockwise turn when the rod runs along +y with the pin on its +x side (``sign``).

    While a bend is being formed, the ``hold`` joints either side of the wrap do not yield: on a real table the roller
    presses the bar against the pin there, so the pin and roller take the bending moment and the straight parts beside
    the bend stay straight. They still bend elastically.

    WHY (2026-11-09, sim trace): for the third bend of a stirrup only 50 mm of rod is left in the channel, and swinging
    the formed U (three sides, 42 links) bent the rod back 43 degrees between the pin and the next guide pin, so the
    corner came out at 43 degrees instead of 90."""

    rod: str
    lever: str
    pin: tuple[float, float]
    wrap_r: float
    moment: float = 0.8
    springback: float = 3.0
    engage_deg: float = 2.0
    hold: int = 5
    sign: float = -1.0
    lever_yaw0: float = 0.0
    theta_max: float = field(init=False, default=0.0)
    j0: int = field(init=False, default=-1)
    s_t: float = field(init=False, default=0.0)
    base: np.ndarray = field(init=False, default=None)
    bends: int = field(init=False, default=0)

    def clear(self) -> None:
        self.theta_max, self.j0, self.s_t, self.base, self.bends = 0.0, -1, 0.0, None, 0


@dataclass
class Plastic:
    """The rod's plasticity: each joint's drive target (its rest angle) follows the joint once the drive's elastic
    moment passes the rod's yield moment. ``set`` holds the rest angles (rad, joint order of the simulator).

    The yield is judged on the joint angles low-pass filtered over ``tau`` seconds (``smooth``), so that only a
    sustained bending moment bends the rod for good.

    WHY (2026-11-09, sim trace): the yield is reached at only 0.29 degrees of elastic bend (0.8 N m against 159 N m/rad
    a joint). Pushed along the plate, the rod stuck and slipped, and the jolt shook its 0.28 g links past that for a
    few substeps: the free end took a 2-degree set and a finished corner opened 3 degrees, with nothing touching
    them."""

    rod: str
    set: np.ndarray = None
    smooth: np.ndarray = None
    tau: float = 0.02

    def clear(self) -> None:
        self.set = None
        self.smooth = None


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
_RODS: dict[str, Rod] = {}
_HINGES: dict = {}  # Hinge by child
_PLASTIC: dict[str, Plastic] = {}
_BENDERS: dict[str, Bender] = {}  # by rod
_MODELS: list = []
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
        if isinstance(s, Rod):
            _RODS[s.name] = s
            _PLASTIC.setdefault(s.name, Plastic(s.name))
        else:
            _SPECS[s.name] = s


def add_hinge(hinge):
    _HINGES[hinge.child] = hinge
    return hinge


def add_bender(b: Bender) -> Bender:
    _BENDERS[b.rod] = b
    return b


def plastic(name: str) -> Plastic:
    return _PLASTIC[name]


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
        if spec.inertia is not None:
            mass.CreateDiagonalInertiaAttr(Gf.Vec3f(*spec.inertia))
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


def _spawn_rod(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    """Author a Rod as an articulation under ``prim_path``: the root Xform carries the articulation root; one rigid
    link per ``link_len`` with a box collider; a revolute joint about z with an angular drive between each pair."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import bind_physics_material, bind_visual_material
    from isaaclab.sim.utils.stage import get_current_stage
    from pxr import Gf, PhysxSchema, UsdGeom, UsdPhysics

    from yam_arena.station import linear_rgb

    spec = _RODS[cfg.body_name]
    stage = get_current_stage()
    root = UsdGeom.Xform.Define(stage, prim_path)
    root.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(*(translation or (0.0, 0.0, 0.0))))
    x, y, z, w = orientation if orientation is not None else (0.0, 0.0, 0.0, 1.0)
    root.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Quatd(w, x, y, z))
    material = f"{prim_path}/Looks/physics"
    pm = sim_utils.RigidBodyMaterialCfg(static_friction=spec.friction, dynamic_friction=spec.friction)
    pm.func(material, pm)
    looks = {}
    for name, color in (("steel", spec.color), ("chalk", spec.mark_color)):
        path = f"{prim_path}/Looks/{name}"
        vis = sim_utils.PreviewSurfaceCfg(diffuse_color=linear_rgb(color), roughness=0.5,
                                          metallic=0.6 if name == "steel" else 0.0)
        vis.func(path, vis)
        looks[name] = path
    n, L, t = spec.links, spec.link_len, spec.thick
    m = spec.mass / n
    deg = math.pi / 180.0
    for i in range(n):
        lp = f"{prim_path}/link{i:02d}"
        link = UsdGeom.Xform.Define(stage, lp)
        link.AddTranslateOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0.0, (i + 0.5) * L, 0.0))
        UsdPhysics.RigidBodyAPI.Apply(link.GetPrim())
        if i == 0:
            # WHY the first link is the articulation's root (2026-11-09, sim trace): with the root on the parent Xform,
            # PhysX chose a link near the middle as the floating root, and Isaac Lab put that link at the spawn pose:
            # the rod started half its length off.
            UsdPhysics.ArticulationRootAPI.Apply(link.GetPrim())
            art = PhysxSchema.PhysxArticulationAPI.Apply(link.GetPrim())
            art.CreateEnabledSelfCollisionsAttr(False)
            art.CreateSolverPositionIterationCountAttr(spec.solver_iterations)
            art.CreateSolverVelocityIterationCountAttr(4)
        mass = UsdPhysics.MassAPI.Apply(link.GetPrim())
        mass.CreateMassAttr(m)
        if spec.link_inertia is not None:
            mass.CreateDiagonalInertiaAttr(Gf.Vec3f(spec.link_inertia, spec.link_inertia, spec.link_inertia))
        rb = PhysxSchema.PhysxRigidBodyAPI.Apply(link.GetPrim())
        rb.CreateMaxDepenetrationVelocityAttr(0.5)
        geom = UsdGeom.Cube.Define(stage, f"{lp}/geom")
        geom.CreateSizeAttr(1.0)
        geom.AddScaleOp().Set(Gf.Vec3f(t, L, t))
        UsdPhysics.CollisionAPI.Apply(geom.GetPrim())
        cp = PhysxSchema.PhysxCollisionAPI.Apply(geom.GetPrim())
        cp.CreateContactOffsetAttr(spec.contact_offset)
        cp.CreateRestOffsetAttr(0.0)
        bind_physics_material(f"{lp}/geom", material, stage=stage)
        bind_visual_material(f"{lp}/geom", looks["steel"], stage=stage)
        for k, sm in enumerate(spec.marks):
            if i * L <= sm < (i + 1) * L:
                mk = UsdGeom.Cube.Define(stage, f"{lp}/mark{k}")
                mk.CreateSizeAttr(1.0)
                mk.AddTranslateOp().Set(Gf.Vec3d(0.0, sm - (i + 0.5) * L, 0.0))
                mk.AddScaleOp().Set(Gf.Vec3f(t * 1.15, 0.0015, t * 1.15))
                bind_visual_material(f"{lp}/mark{k}", looks["chalk"], stage=stage)
        if i > 0:
            j = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/joint{i:02d}")
            j.CreateBody0Rel().SetTargets([f"{prim_path}/link{i - 1:02d}"])
            j.CreateBody1Rel().SetTargets([lp])
            j.CreateAxisAttr("Z")
            j.CreateLocalPos0Attr(Gf.Vec3f(0.0, L / 2, 0.0))
            j.CreateLocalPos1Attr(Gf.Vec3f(0.0, -L / 2, 0.0))
            drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
            drive.CreateTypeAttr("force")
            drive.CreateStiffnessAttr(spec.stiffness * deg)  # USD angular drives are per degree
            drive.CreateDampingAttr(spec.damping * deg)
            drive.CreateTargetPositionAttr(0.0)
    return root.GetPrim()


def _arena_rod(spec: Rod):
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.utils.pose import Pose

    @configclass
    class RodSpawnerCfg(SpawnerCfg):
        func: object = clone(_spawn_rod)
        body_name: str = ""

    # Isaac Lab puts the root link (the first) at the initial pose: its centre is half a link along the rod.
    c, sn = math.cos(spec.yaw), math.sin(spec.yaw)
    half = spec.link_len / 2
    first = (spec.pos[0] - sn * half, spec.pos[1] + c * half, spec.pos[2])
    return Object(
        name=spec.name,
        object_type=ObjectType.ARTICULATION,
        spawner_cfg=RodSpawnerCfg(body_name=spec.name),
        initial_pose=Pose(position_xyz=first, rotation_xyzw=yaw_quat(spec.yaw)),
    )


def _rod_state(environment, spec: Rod) -> RodState:
    base = environment.unwrapped
    art = base.scene.articulations[spec.name]
    origin = _np(base.scene.env_origins)[0]
    names = list(art.body_names)
    poses = _np(art.data.body_link_pose_w)[0]
    vels = _np(art.data.body_link_vel_w)[0]
    order = [names.index(f"link{i:02d}") for i in range(spec.links)]
    poses, vels = poses[order], vels[order]
    jnames = list(art.joint_names)
    q = _np(art.data.joint_pos)[0]
    angles = np.array([q[jnames.index(f"joint{i:02d}")] for i in range(1, spec.links)])
    centres = poses[:, :3] - origin
    axes = np.array([quat_to_matrix(p[3:7])[:, 1] for p in poses])
    half = spec.link_len / 2
    points = np.vstack([centres - axes * half, centres[-1:] + axes[-1:] * half])
    lo = points.min(axis=0) - spec.thick / 2
    hi = points.max(axis=0) + spec.thick / 2
    return RodState(pos=centres[0] - axes[0] * half, quat_xyzw=poses[0, 3:7], lin_vel=vels[0, :3], ang_vel=vels[0, 3:6],
                    low=lo, high=hi, points=points, angles=angles, speeds=np.linalg.norm(vels[:, :3], axis=1))


def _bend(b: Bender, spec: Rod, pl: Plastic, lever, art, base) -> None:
    """One substep of the bender: see :class:`Bender`."""
    import warp as wp

    pose = _np(lever.data.root_link_pose_w)[0]
    r = quat_to_matrix(pose[3:7])
    theta = math.atan2(r[1, 0], r[0, 0]) - b.lever_yaw0
    theta = (theta + math.pi) % (2 * math.pi) - math.pi
    b.theta = theta
    engage = math.radians(b.engage_deg)
    torque = 0.0
    if theta < engage:
        if b.j0 >= 0:  # a bend just finished: it keeps theta_max less the springback
            b.j0, b.base, b.theta_max = -1, None, 0.0
            b.bends += 1
        return
    if b.j0 < 0:  # a bend starts at the rod's tangent point on the pin: the centreline point nearest the pin's axis
        poses = _np(art.data.body_link_pose_w)[0]
        names = list(art.body_names)
        order = [names.index(f"link{i:02d}") for i in range(spec.links)]
        pin = np.array(b.pin) + _np(base.scene.env_origins)[0][:2]
        best = (math.inf, 0.0)
        for i, k in enumerate(order):
            c, u = poses[k, :2], quat_to_matrix(poses[k, 3:7])[:2, 1]
            t = float(np.clip(np.dot(pin - (c - u * spec.link_len / 2), u), 0.0, spec.link_len))
            d = float(np.linalg.norm(c - u * spec.link_len / 2 + u * t - pin))
            if d < best[0]:
                best = (d, i * spec.link_len + t)
        b.s_t = best[1]  # arc length from the free end
        b.j0 = max(0, min(spec.links - 2, int(round(b.s_t / spec.link_len)) - 1))
        b.base = pl.set.copy()
        b.theta_max = 0.0
    forming = theta > b.theta_max
    b.theta_max = max(b.theta_max, theta)
    sb = math.radians(b.springback)
    hold = b.theta_max if theta >= b.theta_max - sb else b.theta_max - sb
    # the wrapped arc runs from the tangent point towards the free end, hold * wrap_r long; each joint takes the part
    # of it within half a link either side (so a bend can start anywhere along a link, not only at a joint)
    lo_arc, L = b.s_t - hold * b.wrap_r, spec.link_len
    for j in range(spec.links - 1):
        sj = (j + 1) * L
        over = min(sj + L / 2, b.s_t) - max(sj - L / 2, lo_arc)
        if over > 0.0:
            pl.set[j] = b.base[j] + b.sign * over / b.wrap_r
    if forming:
        torque = -b.moment
    if torque:
        f = np.zeros(3, dtype=np.float32)
        t = np.array([0.0, 0.0, torque], dtype=np.float32)
        lever.root_view.apply_forces_and_torques_at_position(
            force_data=wp.array(f, dtype=wp.float32, device=base.device),
            torque_data=wp.array(t, dtype=wp.float32, device=base.device),
            position_data=None, indices=lever._ALL_INDICES, is_global=True)


def _install_rod(environment, spec: Rod) -> None:
    """Set each joint's stiffness and damping (SI) and hook the rod's write so the plastic rest angles follow."""
    import warp as wp

    base = environment.unwrapped
    art = base.scene.articulations[spec.name]
    n = spec.links - 1
    art.write_joint_stiffness_to_sim_index(stiffness=spec.stiffness)
    art.write_joint_damping_to_sim_index(damping=spec.damping)
    if getattr(art, "_rodkit_hooked", False):
        return
    pl = _PLASTIC[spec.name]
    bender = _BENDERS.get(spec.name)
    lever_obj = base.scene.rigid_objects[bender.lever] if bender is not None else None
    orig = art.write_data_to_sim
    dev = base.device

    def write():
        orig()
        q = _np(art.root_view.get_dof_positions())[0]
        if pl.set is None or len(pl.set) != n:
            pl.set = q.copy()
        if pl.smooth is None or len(pl.smooth) != n:
            pl.smooth = q.copy()
        pl.smooth += (q - pl.smooth) * min(1.0, float(base.physics_dt) / pl.tau)
        qs = pl.smooth
        e = qs - pl.set
        lim = spec.yield_moment / spec.stiffness
        over = np.abs(e) > lim
        if bender is not None and bender.j0 >= 0:  # forming: the roller holds the rod beside the wrap (see Bender)
            wrapped = int(math.ceil(math.radians(120.0) / (spec.link_len / bender.wrap_r)))  # any bend's wrap
            over[max(0, bender.j0 - wrapped - bender.hold + 1):bender.j0 + bender.hold + 1] = False
        if over.any():
            pl.set[over] = qs[over] - np.sign(e[over]) * lim
        if bender is not None:
            _bend(bender, spec, pl, lever_obj, art, base)
        targets = wp.array(pl.set[None, :].astype(np.float32), dtype=wp.float32, device=dev)
        art.root_view.set_dof_position_targets(targets, art._ALL_INDICES)

    art.write_data_to_sim = write
    art._rodkit_hooked = True


def _t(x):
    return getattr(x, "torch", x)


def _np(x) -> np.ndarray:
    x = _t(x)
    if hasattr(x, "detach"):
        return x.detach().cpu().numpy().astype(np.float64)
    return np.asarray(x.numpy(), dtype=np.float64)  # a warp array


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
    for m in _MODELS:
        if hasattr(m, "dt"):
            m.dt = float(base.physics_dt)
    if getattr(anchor, "_rodkit_hooked", False):
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
            # inside warp's type_ctype); see our paint-tin task's helper.
            o.root_view.apply_forces_and_torques_at_position(
                force_data=wp.array(f.astype(np.float32), dtype=wp.float32, device=dev),
                torque_data=wp.array(t.astype(np.float32), dtype=wp.float32, device=dev),
                position_data=None,
                indices=o._ALL_INDICES,
                is_global=True,
            )

    anchor.write_data_to_sim = write
    anchor._rodkit_hooked = True


def _patch_scene(scene) -> None:
    if getattr(scene, "_rodkit_patched", False):
        return
    orig_task_object, orig_states = scene.task_object, scene.object_states

    def task_object(spec):
        if isinstance(spec, Rod):
            _RODS[spec.name] = spec
            _PLASTIC.setdefault(spec.name, Plastic(spec.name))
            return _arena_rod(spec)
        if isinstance(spec, Body):
            _SPECS[spec.name] = spec
            return _arena_object(spec)
        return orig_task_object(spec)

    def object_states(environment, names):
        plain = [n for n in names if n not in _SPECS and n not in _RODS]
        out = orig_states(environment, plain) if plain else {}
        out.update(_states(environment, [n for n in names if n in _SPECS]))
        out.update({n: _rod_state(environment, _RODS[n]) for n in names if n in _RODS})
        return {n: out[n] for n in names}

    scene.task_object = task_object
    scene.object_states = object_states
    scene._rodkit_patched = True


def _patch_embodiment(ie) -> None:
    cls = ie.YamArenaEmbodiment
    if getattr(cls, "_rodkit_patched", False):
        return
    base_reset = cls.reset

    def reset(self, *a, **k):
        # WHY: the force models' state and any latched violations must be cleared before the env reset and its
        # settle steps run, or a rod bent in the last trial would start this one bent.
        for m in _MODELS:
            m.clear()
        for pl in _PLASTIC.values():
            pl.clear()
        for bd in _BENDERS.values():
            bd.clear()
        for fn in _RESET:
            fn()
        first = getattr(self, "_env", None) is None
        if not first:
            _install_forces(self._env)
        obs = base_reset(self, *a, **k)
        _install_forces(self._env)  # the first reset builds the env
        for spec in _RODS.values():
            _install_rod(self._env, spec)
        if first and (_MODELS or _RODS):
            # WHY reset twice the first time (2026-11-07, glass-pane task's sim trace): the first reset builds the env
            # and runs its settle steps before the force hooks exist, so force models miss them.
            for m in _MODELS:
                m.clear()
            for pl in _PLASTIC.values():
                pl.clear()
            obs = base_reset(self, *a, **k)
        _ENV["env"] = self._env
        for m in _MODELS:  # whatever the settle steps did, every trial starts clean
            m.clear()
        for pl in _PLASTIC.values():
            pl.clear()
        for bd in _BENDERS.values():
            bd.clear()
        for fn in _RESET:
            fn()
        return obs

    cls.reset = reset
    cls._rodkit_patched = True


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
