"""camkit: compound rigid bodies and a shelf joined to a cabinet side by cam-and-bolt connectors, for a yam_arena task
file.

Written for the cam-lock task in this folder, from the helpers of our earlier tasks. Importing it is cheap
(NumPy only): everything that needs Isaac Sim is imported lazily, inside hooks that run once the simulator has
started.

What it adds to a plain task file:

- :class:`Body`: one rigid body (or a fixed fixture, without a mass) made of several boxes and cylinders; a colour
  with a fourth (alpha) component below 1 is drawn see-through (glass).
- :class:`Slider` and :class:`Hinge`: prismatic and revolute joints between two bodies.
- :class:`CamLockJoint`: a shelf joined to an upright side by cam-and-bolt connectors turned with screwdrivers (our
  own model: the bolts and their holes, the cams, the screwdrivers in the cams' recesses), the cams drawn by bodies we
  pose.
- Models run before every physics substep (240 Hz), so a quick motion cannot skip them.
- :func:`finger_points`: the grippers' collision-hull vertices, for scoring whether a gripper touches a part.
- :func:`on_reset`: functions to run at every episode reset, e.g. to clear latched scoring state.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import math
import os
import sys
from dataclasses import dataclass, field

import numpy as np

from yam_arena.task_file import Box, ObjectState

__all__ = [
    "Shape", "box", "cyl", "ring", "prism12", "Body", "body", "BodyState", "CamLockJoint", "Hinge", "add_hinge",
    "no_gravity", "add_model", "on_reset", "register", "finger_points", "install",
]

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


@dataclass(frozen=True)
class Hinge:
    """A revolute joint between two bodies (the child turns on the parent), authored as a PhysX joint when the child
    is spawned: ``axis`` in the parent's frame through ``anchor``; ``damping`` (N m s/rad) brakes it, ``stiffness``
    (N m/rad) springs it towards ``target``, ``lower``/``upper`` (rad) limit it. The two bodies do not collide."""

    parent: str
    child: str
    anchor: tuple[float, float, float]
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    damping: float = 0.0
    stiffness: float = 0.0  # N m/rad towards ``target`` (rad): a spring
    target: float = 0.0
    lower: float | None = None  # rad
    upper: float | None = None


@dataclass(frozen=True)
class Slider:
    """A prismatic joint between two bodies (the child slides along the parent), authored as a PhysX joint when the
    child is spawned. ``axis`` is in the parent's frame, through ``anchor``; the child can move from ``lower`` to
    ``upper`` (m) along it from where it is spawned (its offset 0 there). The two bodies do not collide."""

    parent: str
    child: str
    anchor: tuple[float, float, float]
    axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    lower: float = -0.1
    upper: float = 0.1


def _wrap_angle(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def _torque(acc: dict, name: str, torque) -> None:
    """Add a pure torque (N m, world) to body ``name``."""
    f, t = acc.setdefault(name, [np.zeros(3), np.zeros(3)])
    t += torque


class CamLockJoint:
    """A shelf joined to an upright side by two cam-and-bolt connectors: our own model (the bolts, the holes, the cams).

    The shelf (body ``shelf``) lies flat; its origin is at its middle, its x axis points to its joining edge (``half``
    m from the origin), z up. The side's face is at world x ``face_x``; for each connector ``k`` a bolt stands out of it
    ``bolt_len`` along -x at world y ``bolt_y[k]`` (the shelf's mid-thickness), with a cam in the shelf's top face at
    ``cam_xy[k]`` (shelf frame). ``gap[k]`` is the distance from the face to the shelf's edge at that bolt.

    - Pushed onto the bolts, the shelf's edge holes take them only if they line up (within ``align`` across, and the
      shelf square to the face within ``align_deg``); otherwise the edge stands on the bolts' ends.
    - A cam's angle ``phi[k]`` is measured clockwise (seen from above) from open, its opening facing the bolt. Open
      (within 30 degrees), the bolt's head goes into it and the shelf comes up to ``g_home`` of the face; closed, the
      head meets the cam's side ``g_closed`` short, and the shelf stops there however hard it is pushed (breaking the
      board out takes far more force than these arms have; nothing breaks here).
    - With the head in it, the cam turned clockwise draws the bolt in (an eccentric: the gap closes as
      (1 + cos phi) / 2 of ``g_home``, shut at 180 degrees); past 30 degrees it holds the head, and the joint cannot be
      pulled apart. Its stops are at -10 and 200 degrees. With no bolt in it, it turns freely.
    - A screwdriver (bodies ``drivers``; origin at its tip, z up its axis) lowered onto a cam within ``capture`` of its
      middle is guided into the cross recess (gently: ``k_guide`` N/m), and within ``engage`` it is in: from then on the cam turns with it, its
      tip is held on the cam's axis and it stands upright by itself, until it is lifted off.
    - ``cams`` (bodies we pose) draw the cams. ``holes``: the screwdrivers' holders, whose chamfered mouths guide a
      blade's tip within ``mouth`` into the hole."""

    def __init__(self, shelf, drivers, cams, face_x, half, bolt_y, cam_xy, top, bolt_len=0.025, g_home=1.5e-3,
                 g_closed=9.0e-3, align=2.5e-3, align_deg=2.5, k_stop=5000.0, capture=4.0e-3,
                 engage=2.0e-3, k_side=2000.0, k_guide=400.0, k_up=0.3, phi0=(0.0, 0.0), holes=(), mouth=6.0e-3):
        self.shelf, self.drivers, self.cams = shelf, list(drivers), list(cams)
        self.face_x, self.half, self.top = float(face_x), float(half), float(top)
        self.bolt_y = [float(v) for v in bolt_y]
        self.cam_xy = [np.asarray(c, float) for c in cam_xy]
        self.bolt_len, self.g_home, self.g_closed = bolt_len, g_home, g_closed
        self.align, self.align_rad, self.k_stop = align, math.radians(align_deg), k_stop
        self.capture, self.engage, self.k_side, self.k_guide, self.k_up = capture, engage, k_side, k_guide, k_up
        self.phi0 = tuple(float(p) for p in phi0)
        self.holes = [np.asarray(h, float) for h in holes]  # the holders' holes: (x, y, top z), world
        self.mouth = mouth
        self.lo, self.hi = math.radians(-10.0), math.radians(200.0)
        self.dt = 1.0 / 240.0
        self.origin = np.zeros(3)
        self.poses = {}
        self.clear()

    @property
    def bodies(self) -> list[str]:
        return [self.shelf] + self.drivers + self.cams

    def clear(self) -> None:
        n = len(self.cams)
        self.phi = list(self.phi0)
        self.entered = [False] * n  # the bolt's shank in its edge hole
        self.bolt_in = [False] * n  # the bolt's head in the cam
        self.gap = [1.0] * n
        self.push = [0.0] * n  # the force pushing the edge on at that bolt (N)
        self.on = {d: None for d in self.drivers}  # the cam each screwdriver is in
        self._yaw = {d: 0.0 for d in self.drivers}

    @staticmethod
    def _yaw_of(rot) -> float:
        return math.atan2(rot[1, 0], rot[0, 0])

    def is_open(self, k) -> bool:
        return abs(_wrap_angle(self.phi[k])) < math.radians(30.0)

    def locked(self, k) -> bool:
        return self.bolt_in[k] and self.phi[k] >= math.radians(120.0)

    def target_gap(self, k) -> float:
        return self.g_home * (1.0 + math.cos(min(max(self.phi[k], 0.0), math.pi))) / 2.0

    def update(self, frames: dict, acc: dict) -> None:
        fs = frames.get(self.shelf)
        if fs is None:
            return
        sx = fs.rot[:, 0]
        skew = math.atan2(sx[1], sx[0])  # the shelf's turn from square to the face
        for k in range(len(self.cams)):
            yk = self.bolt_y[k]
            mid = fs.world(np.array([self.half, 0.0, 0.0])) - self.origin  # the middle of the joining edge
            along = (yk - mid[1]) / max(1e-6, float(fs.rot[1, 1]))  # how far along the edge its point on the bolt's line
            ex = float(mid[0] + along * fs.rot[0, 1])
            hole_y = float(fs.world(np.array([self.half, self.cam_xy[k][1], 0.0]))[1] - self.origin[1])
            edge = mid
            gap = self.face_x - ex
            self.gap[k] = gap
            at = np.array([ex, yk, edge[2]]) + self.origin
            stop = None
            if gap > self.bolt_len + 1e-3:
                self.entered[k] = self.bolt_in[k] = False
            elif not self.entered[k]:
                if abs(hole_y - yk) <= self.align and abs(skew) <= self.align_rad:
                    self.entered[k] = True
                else:
                    stop = self.bolt_len  # the edge stands on the bolt's end
            if self.entered[k] and not self.bolt_in[k]:
                if self.is_open(k):
                    stop = self.g_home
                    if gap <= self.g_home + 1.0e-3:
                        self.bolt_in[k] = True
                        self.phi[k] = _wrap_angle(self.phi[k])
                else:
                    stop = self.g_closed  # the head meets the cam's side
            f = 0.0
            if self.bolt_in[k]:
                target = self.target_gap(k)
                if gap > self.g_closed:
                    self.bolt_in[k] = False
                elif self.phi[k] >= math.radians(30.0):  # the cam holds the head: drawn to the target
                    f = self.k_stop * (gap - target) - 5.0 * float(fs.vel[0])
                elif gap < target:
                    f = self.k_stop * (gap - target) - 5.0 * float(fs.vel[0])
            elif stop is not None and gap < stop:
                f = self.k_stop * (gap - stop) - 5.0 * float(fs.vel[0])  # < 0: pushed back
            self.push[k] = max(0.0, -f) if not self.bolt_in[k] else 0.0
            if f:
                _wrench(acc, self.shelf, fs, np.array([f, 0.0, 0.0]), at)
        # the screwdrivers
        for name in self.drivers:
            fd = frames.get(name)
            if fd is None:
                continue
            tip = fd.pos
            yaw = self._yaw_of(fd.rot)
            cams = [fs.world(np.array([c[0], c[1], self.top])) for c in self.cam_xy]
            k = self.on[name]
            if k is None:
                for h in self.holes:  # a holder's chamfered mouth guides the blade's tip into its hole
                    d = float(np.hypot(*(tip[:2] - h[:2])))
                    if d <= self.mouth and -0.04 < tip[2] - h[2] < 8e-3:
                        side = self.k_guide * (h[:2] - tip[:2]) - 0.5 * np.asarray(fd.vel[:2])
                        _wrench(acc, name, fd, np.array([side[0], side[1], 0.0]), tip)
                for j, c in enumerate(cams):
                    d = float(np.hypot(*(tip[:2] - c[:2])))
                    dz = float(tip[2] - c[2])
                    if d <= self.engage and dz < 1.5e-3 and all(self.on[o] != j for o in self.drivers):
                        self.on[name], self._yaw[name] = j, yaw
                        k = j
                        break
                    if d <= self.capture and dz < 3e-3:  # the tip in the recess's cone: guided to the middle
                        side = self.k_guide * (c[:2] - tip[:2]) - 0.5 * np.asarray(fd.vel[:2])
                        _wrench(acc, name, fd, np.array([side[0], side[1], 0.0]), tip)
            if k is not None:
                c = cams[k]
                d = float(np.hypot(*(tip[:2] - c[:2])))
                if tip[2] - c[2] > 3e-3 or d > self.engage + 3e-3:
                    self.on[name] = None  # lifted off
                    continue
                side = self.k_side * (c[:2] - tip[:2]) - 2.0 * np.asarray(fd.vel[:2])
                _wrench(acc, name, fd, np.array([side[0], side[1], 0.0]), tip)
                up = fd.rot[:, 2]  # stands upright in the recess
                level = self.k_up * np.cross(up, np.array([0.0, 0.0, 1.0]))
                _torque(acc, name, level - 0.003 * (fd.spin - up * np.dot(fd.spin, up)))
                turn = -_wrap_angle(yaw - self._yaw[name])  # clockwise positive
                self._yaw[name] = yaw
                new = self.phi[k] + turn
                if self.bolt_in[k]:
                    held = min(max(new, self.lo), self.hi)
                    if held != new:  # at a stop: the cam holds the screwdriver
                        _torque(acc, name, np.array([0.0, 0.0, 2.0 * (new - held) - 0.02 * fd.spin[2]]))
                    new = held
                self.phi[k] = new
        self._draw(fs)

    def _draw(self, fs) -> None:
        self.poses = {}
        base = math.atan2(fs.rot[1, 0], fs.rot[0, 0])
        for k, name in enumerate(self.cams):
            c = fs.world(np.array([self.cam_xy[k][0], self.cam_xy[k][1], self.top]))
            self.poses[name] = (c, yaw_quat(base - self.phi[k]))


def add_slider(s: Slider) -> Slider:
    _SLIDERS[s.child] = s
    return s


def no_gravity(name: str) -> None:
    _NO_GRAVITY.add(name)


_SPECS: dict[str, Body] = {}
_SLIDERS: dict = {}  # Slider by child
_HINGES: dict = {}  # Hinge by child
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
    slider = _SLIDERS.get(cfg.body_name)
    if slider is not None:
        _author_slider(stage, prim_path, slider)
    hinge = _HINGES.get(cfg.body_name)
    if hinge is not None:
        _author_hinge(stage, prim_path, hinge)
    return prim


def _author_slider(stage, prim_path: str, sl: Slider) -> None:
    """A PhysX prismatic joint from the parent (a sibling prim) to this child, along ``axis`` (the parent's frame)
    through ``anchor``. The child's frame is placed where it is spawned, so both joint frames are given in their
    own bodies' frames from the spawn poses."""
    from pxr import Gf, UsdPhysics

    parent_path = prim_path.rsplit("/", 1)[0] + "/" + sl.parent
    pspec, cspec = _SPECS[sl.parent], _SPECS[sl.child]
    rp = quat_to_matrix(pspec.quat) if pspec.quat else quat_to_matrix(yaw_quat(pspec.yaw))
    rc = quat_to_matrix(cspec.quat) if cspec.quat else quat_to_matrix(yaw_quat(cspec.yaw))
    anchor_w = np.asarray(pspec.pos, float) + rp @ np.asarray(sl.anchor, float)
    local1 = rc.T @ (anchor_w - np.asarray(cspec.pos, float))
    rel = rc.T @ rp  # the parent's frame seen from the child's
    q1 = matrix_to_quat_rel(rel)
    j = UsdPhysics.PrismaticJoint.Define(stage, f"{prim_path}/slider")
    j.CreateBody0Rel().SetTargets([parent_path])
    j.CreateBody1Rel().SetTargets([prim_path])
    a = np.asarray(sl.axis, float)
    j.CreateAxisAttr({0: "X", 1: "Y", 2: "Z"}[int(np.argmax(np.abs(a)))])
    j.CreateLocalPos0Attr(Gf.Vec3f(*map(float, sl.anchor)))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(*map(float, local1)))
    j.CreateLocalRot1Attr(Gf.Quatf(float(q1[3]), float(q1[0]), float(q1[1]), float(q1[2])))
    j.CreateLowerLimitAttr(float(sl.lower))
    j.CreateUpperLimitAttr(float(sl.upper))
    j.CreateCollisionEnabledAttr(False)


def _author_hinge(stage, prim_path: str, h: "Hinge") -> None:
    """A PhysX revolute joint from the parent (a sibling prim) to this child about ``axis`` (the parent's frame)
    through ``anchor``, with an angular drive (a spring and a brake) and limits."""
    from pxr import Gf, UsdPhysics

    parent_path = prim_path.rsplit("/", 1)[0] + "/" + h.parent
    pspec, cspec = _SPECS[h.parent], _SPECS[h.child]
    rp = quat_to_matrix(pspec.quat) if pspec.quat else quat_to_matrix(yaw_quat(pspec.yaw))
    rc = quat_to_matrix(cspec.quat) if cspec.quat else quat_to_matrix(yaw_quat(cspec.yaw))
    anchor_w = np.asarray(pspec.pos, float) + rp @ np.asarray(h.anchor, float)
    local1 = rc.T @ (anchor_w - np.asarray(cspec.pos, float))
    q1 = matrix_to_quat_rel(rc.T @ rp)
    j = UsdPhysics.RevoluteJoint.Define(stage, f"{prim_path}/hinge")
    j.CreateBody0Rel().SetTargets([parent_path])
    j.CreateBody1Rel().SetTargets([prim_path])
    axis = {0: "X", 1: "Y", 2: "Z"}[int(np.argmax(np.abs(np.asarray(h.axis, float))))]
    j.CreateAxisAttr(axis)
    j.CreateLocalPos0Attr(Gf.Vec3f(*map(float, h.anchor)))
    j.CreateLocalRot0Attr(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    j.CreateLocalPos1Attr(Gf.Vec3f(*map(float, local1)))
    j.CreateLocalRot1Attr(Gf.Quatf(float(q1[3]), float(q1[0]), float(q1[1]), float(q1[2])))
    j.CreateCollisionEnabledAttr(False)
    if h.lower is not None:
        j.CreateLowerLimitAttr(math.degrees(h.lower))  # USD revolute limits are in degrees
    if h.upper is not None:
        j.CreateUpperLimitAttr(math.degrees(h.upper))
    drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(float(h.stiffness) * math.pi / 180.0)  # USD angular drives are per degree
    drive.CreateDampingAttr(float(h.damping) * math.pi / 180.0)
    drive.CreateTargetPositionAttr(math.degrees(h.target))
    drive.CreateTargetVelocityAttr(0.0)


def add_hinge(h: "Hinge") -> "Hinge":
    _HINGES[h.child] = h
    return h


def matrix_to_quat_rel(r: np.ndarray) -> tuple[float, float, float, float]:
    """xyzw quaternion of a rotation matrix."""
    tr = r[0, 0] + r[1, 1] + r[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        return ((r[2, 1] - r[1, 2]) / s, (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s, 0.25 * s)
    i = int(np.argmax([r[0, 0], r[1, 1], r[2, 2]]))
    if i == 0:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
        return (0.25 * s, (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s, (r[2, 1] - r[1, 2]) / s)
    if i == 1:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
        return ((r[0, 1] + r[1, 0]) / s, 0.25 * s, (r[1, 2] + r[2, 1]) / s, (r[0, 2] - r[2, 0]) / s)
    s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
    return ((r[0, 2] + r[2, 0]) / s, (r[1, 2] + r[2, 1]) / s, 0.25 * s, (r[1, 0] - r[0, 1]) / s)


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


def finger_points(robots=ROBOTS) -> np.ndarray:
    """World positions (n, 3) of the vertices of the arms' gripper collision hulls (palm and fingers; the URDF's
    CoACD hulls), for a scorer to tell whether a gripper still touches a part. Empty before the first reset and
    offline."""
    environment = _ENV["env"]
    if environment is None:
        return np.zeros((0, 3))
    from yam_arena.analysis import world_hulls
    from yam_arena.runtime import link_poses

    scene = environment.unwrapped.scene
    pts = [h.vertices for name in robots for _, h in world_hulls(link_poses(scene[name]), list(FINGER_LINKS))]
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
    objs = {n: base.scene.rigid_objects[n] for n in names if _SPECS[n].mass is not None}
    origin = _np(base.scene.env_origins)[0][:3]
    fixed = {}  # fixtures a model names (no mass): their frames never change, and nothing pushes them
    for n in names:
        sp = _SPECS[n]
        if sp.mass is None:
            rot = quat_to_matrix(sp.quat) if sp.quat else quat_to_matrix(yaw_quat(sp.yaw))
            pos = np.asarray(sp.pos, float) + origin
            fixed[n] = Frame(pos=pos, rot=rot, vel=np.zeros(3), spin=np.zeros(3),
                             com=pos + rot @ np.asarray(sp.com, float))
    anchor = objs[sorted(objs)[0]]
    for m in _MODELS:
        if hasattr(m, "dt"):
            m.dt = float(base.physics_dt)
        if hasattr(m, "origin"):
            m.origin = _np(base.scene.env_origins)[0][:3]
    if getattr(anchor, "_camkit_hooked", False):
        return
    orig = anchor.write_data_to_sim

    def write():
        orig()
        frames = {n: _frame(o, _SPECS[n]) for n, o in objs.items()}
        frames.update(fixed)
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
            if n not in objs or (not f.any() and not t.any()):
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
    anchor._camkit_hooked = True


def _patch_scene(scene) -> None:
    if getattr(scene, "_camkit_patched", False):
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
    scene._camkit_patched = True


def _patch_embodiment(ie) -> None:
    cls = ie.YamArenaEmbodiment
    if getattr(cls, "_camkit_patched", False):
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
    cls._camkit_patched = True


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
