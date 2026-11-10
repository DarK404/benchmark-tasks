"""tapekit: compound rigid bodies and thread-seal tape wound onto a pipe thread, for a yam_arena task file.

Written for the thread-taping task in this folder, from the helpers of our earlier tasks. Importing it is cheap
(NumPy only): everything that needs Isaac Sim is imported lazily, inside hooks that run once the simulator has
started.

What it adds to a plain task file:

- :class:`Body`: one rigid body (or a fixed fixture, without a mass) made of several boxes and cylinders; a colour
  with a fourth (alpha) component below 1 is drawn see-through (glass).
- :class:`TapeWrap`: thread-seal tape wound from a roll onto a pipe thread (our own path model), drawn by bodies we
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

__all__ = ["Shape", "box", "cyl", "ring", "prism12", "Body", "body", "BodyState", "TapeWrap", "Hinge", "add_hinge", "no_gravity",
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


@dataclass(frozen=True)
class Hinge:
    """A revolute joint between two bodies (the child turns on the parent), authored as a PhysX joint when the child
    is spawned: ``axis`` in the parent's frame through ``anchor``; ``damping`` (N m s/rad) brakes it. The two bodies
    do not collide."""

    parent: str
    child: str
    anchor: tuple[float, float, float]
    axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    damping: float = 0.0


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


class TapeWrap:
    """Thread-seal tape wound from a roll onto an upright threaded pipe: our own model (PhysX has no tape).

    The pipe's axis is upright through ``axis_xy`` (env frame), its thread of radius ``r_pipe`` from ``z_shoulder``
    to ``z_top``. The roll (body ``roll``, radius ``roll_r``, its tape ``width`` wide) turns on a spindle in a
    dispenser (body ``dispenser``). The tape starts when the roll is touched to the thread: its end sticks there (the
    anchor). From then on it is wound by the angle the roll travels round the pipe's axis (``wound``, radians,
    positive clockwise seen from above, i.e. from the thread's end), each part laid at the roll's height while the roll
    is level with the thread; carried back, it unwinds.

    The tape comes off the roll only as the roll turns on its spindle (clockwise seen from above): ``paid`` is that
    turning times its radius. The tape needed is the length wound plus the free run from the pipe to the roll; what is
    needed beyond what has been paid out stretches the tape, which pulls on the roll's rim with ``k_tension`` N/m,
    along the run (this tension turns the roll, against its spindle's brake). Stretched ``break_stretch``, the tape
    tears: winding steadily it pays out under a light pull; yanked, it outruns the braked roll and tears, and a roll
    that cannot turn tears it at once. Torn before the first half turn has bound its end, it falls off. The torn tail
    stands loose until a hand presses it down onto the wrap (a gripper's hull within ``press_reach`` of it).

    The wound tape and the run from the pipe to the roll are drawn by moving bodies we pose (``segments``: one per
    quarter turn in laying order, ``span``, ``tail``)."""

    def __init__(self, roll, dispenser, segments, span, tail, axis_xy, r_pipe, z_shoulder, z_top, roll_r, width,
                 hidden_z, touch_gap=0.003, press_reach=0.008, k_tension=300.0, break_stretch=0.008):
        self.roll, self.dispenser = roll, dispenser
        self.segments, self.span, self.tail = list(segments), span, tail
        self.axis = np.asarray(axis_xy, float)
        self.r_pipe, self.z_shoulder, self.z_top = float(r_pipe), float(z_shoulder), float(z_top)
        self.roll_r, self.width, self.hidden_z = float(roll_r), float(width), float(hidden_z)
        self.touch_gap, self.press_reach = touch_gap, press_reach
        self.k_tension, self.break_stretch = float(k_tension), float(break_stretch)
        self.dt = 1.0 / 240.0
        self.origin = np.zeros(3)
        self.poses = {}
        self.clear()

    @property
    def bodies(self) -> list[str]:
        return [self.roll, self.dispenser, *self.segments, self.span, self.tail]

    def clear(self) -> None:
        self.anchored = False  # the tape's end is stuck to the thread
        self.secured = False  # the first half turn has passed over the end: the tape holds itself
        self.torn = False
        self.tail_down = False
        self.anchor = 0.0  # the end's angle round the axis (rad, from +x, counter-clockwise from above)
        self.wound = 0.0
        self.samples: list[tuple[float, float]] = []  # (wound angle, height) as the tape was laid
        self.paid = 0.0  # tape off the roll (m): the roll's clockwise turning on its spindle times its radius
        self.stretch = 0.0
        self.tension = 0.0
        self.peak_tension = 0.0
        self._paid0 = self._run0 = 0.0
        self._spin = None
        self._last = None
        self._tick = 0
        self._fingers = np.zeros((0, 3))
        self._shown: dict = {}

    # -- geometry --------------------------------------------------------------------------------------------------
    def turns(self) -> float:
        return self.wound / (2 * math.pi)

    def bands(self) -> list[tuple[float, float]]:
        """The (bottom, top) heights of the tape laid on the thread."""
        return [(z - self.width / 2, z + self.width / 2) for _, z in self.samples]

    def _hands(self, roll_pos):
        """Gripper hull points that are not round the roll (the hand carrying it), read every fourth substep."""
        self._tick += 1
        if self._tick % 4 == 1:
            pts = finger_points()
            if len(pts):
                pts = pts[np.linalg.norm(pts - roll_pos, axis=1) > self.roll_r + 0.012]
            self._fingers = pts
        return self._fingers

    def _point(self, angle, z, r=None):
        r = self.r_pipe if r is None else r
        return np.array([self.axis[0] + r * math.cos(angle), self.axis[1] + r * math.sin(angle), z]) + self.origin

    def run(self, roll_pos, R, psi):
        """The free run of tape: where it leaves the pipe (world), where it leaves the roll (world: the side from
        which pulling turns the roll clockwise seen from above) and its length."""
        sign = 1.0 if self.wound >= 0 else -1.0
        a = self._point(psi + sign * math.acos(min(1.0, self.r_pipe / max(R, 1e-9))), roll_pos[2])
        d = a[:2] - roll_pos[:2]
        dist = float(np.hypot(*d))
        if dist <= self.roll_r:
            return a, roll_pos.copy(), 0.0
        phi, beta = math.atan2(d[1], d[0]), math.acos(self.roll_r / dist)
        for b in (beta, -beta):
            q = roll_pos + self.roll_r * np.array([math.cos(phi + b), math.sin(phi + b), 0.0])
            r_q, f = q - roll_pos, a - q
            if r_q[0] * f[1] - r_q[1] * f[0] < 0:  # pulling here turns the roll clockwise
                return a, q, float(np.linalg.norm(a - q))
        return a, roll_pos.copy(), dist

    # -- the model -------------------------------------------------------------------------------------------------
    def update(self, frames: dict, acc: dict) -> None:
        fr, fd = frames.get(self.roll), frames.get(self.dispenser)
        if fr is None or fd is None:
            return
        rel_rot = fd.rot.T @ fr.rot  # the roll turned on its spindle
        spin = math.atan2(rel_rot[1, 0], rel_rot[0, 0])
        if self._spin is not None:
            self.paid -= _wrap_angle(spin - self._spin) * self.roll_r  # clockwise pays out
        self._spin = spin
        p = fr.pos - self.origin
        rel = p[:2] - self.axis
        R = float(np.hypot(*rel))
        psi = math.atan2(rel[1], rel[0])
        hands = self._hands(fr.pos)
        self.tension = 0.0
        if not self.anchored and not self.torn:
            if (R - self.roll_r - self.r_pipe <= self.touch_gap and self.z_shoulder <= p[2] <= self.z_top):
                self.anchored, self.anchor, self.wound, self._last = True, psi, 0.0, psi
                self.samples = [(0.0, float(p[2]))]
                self._paid0, self._run0 = self.paid, self.run(fr.pos, R, psi)[2]
        elif self.anchored and not self.torn:
            self.wound -= _wrap_angle(psi - self._last)  # clockwise from above is positive
            self._last = psi
            w = abs(self.wound)
            self.secured = self.secured or w >= math.pi
            last_w = abs(self.samples[-1][0])
            if self.wound * self.samples[-1][0] < 0 and w > 0.05:  # back through the start: wound afresh
                self.samples = [(0.0, self.samples[0][1])]
                last_w = 0.0
            # (laid only while the roll is level with the thread: clear above or below it, the run just pulls)
            if (w > last_w + math.radians(5)
                    and self.z_shoulder - self.width / 2 <= p[2] <= self.z_top + self.width / 2):
                self.samples.append((self.wound, float(p[2])))
            while len(self.samples) > 1 and abs(self.samples[-1][0]) > w + math.radians(5):
                self.samples.pop()  # carried back: it unwinds
            a, q, length = self.run(fr.pos, R, psi)
            need = w * self.r_pipe + length - self._run0
            self.stretch = need - (self.paid - self._paid0)
            if self.stretch > self.break_stretch:
                self._tear()
            elif self.stretch > 0.0 and length > 0.0:
                self.tension = self.k_tension * self.stretch
                _wrench(acc, self.roll, fr, self.tension * (a - q) / length, q)
            self.peak_tension = max(self.peak_tension, self.tension)
        elif self.torn and self.anchored and not self.tail_down and len(hands):
            self.tail_down = bool(np.linalg.norm(hands - self.tail_point(), axis=1).min() < self.press_reach)
        self._draw(R, psi, p)

    def tail_point(self) -> np.ndarray:
        """Where the torn tape ends on the thread (world)."""
        return self._point(self.anchor - self.wound, self.samples[-1][1])

    def _tear(self) -> None:
        self.torn = True
        if not self.secured:  # torn before the first half turn bound its end: it falls off
            self.anchored, self.samples, self.wound = False, [], 0.0

    # -- drawing ---------------------------------------------------------------------------------------------------
    def _pose(self, name, pos, yaw) -> None:
        q = yaw_quat(yaw)
        key = tuple(np.round(np.r_[pos, q], 5))
        if self._shown.get(name) != key:
            self._shown[name] = key
            self.poses[name] = (np.asarray(pos, float), q)

    def _hide(self, name) -> None:
        self._pose(name, np.array([self.axis[0], self.axis[1], self.hidden_z]) + self.origin, 0.0)

    def _draw(self, R, psi, p) -> None:
        self.poses = {}
        sign = 1.0 if self.wound >= 0 else -1.0
        w = abs(self.wound) if self.anchored else 0.0
        quarter = math.pi / 2
        for k, name in enumerate(self.segments):
            if self.anchored and w >= (k + 1) * quarter:
                zs = [z for a, z in self.samples if k * quarter <= abs(a) <= (k + 1) * quarter + 0.01] or [
                    self.samples[-1][1]]
                mid = self.anchor - sign * (k + 0.5) * quarter
                self._pose(name, np.array([self.axis[0], self.axis[1], float(np.mean(zs))]) + self.origin, mid)
            else:
                self._hide(name)
        if self.anchored and not self.torn and R > self.r_pipe:
            start, q, _ = self.run(p + self.origin, R, psi)
            self._pose(self.span, start, math.atan2(q[1] - start[1], q[0] - start[0]))
        else:
            self._hide(self.span)
        if self.torn and self.anchored and not self.tail_down:
            end_angle = self.anchor - self.wound
            self._pose(self.tail, self._point(end_angle, self.samples[-1][1]), end_angle - sign * math.pi / 2)
        else:
            self._hide(self.tail)


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
    through ``anchor``, with an angular drive that only damps (``damping`` N m s/rad): a braked spindle."""
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
    drive = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "angular")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(0.0)
    drive.CreateDampingAttr(float(h.damping) * math.pi / 180.0)  # USD angular drives are per degree
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
    if getattr(anchor, "_tapekit_hooked", False):
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
    anchor._tapekit_hooked = True


def _patch_scene(scene) -> None:
    if getattr(scene, "_tapekit_patched", False):
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
    scene._tapekit_patched = True


def _patch_embodiment(ie) -> None:
    cls = ie.YamArenaEmbodiment
    if getattr(cls, "_tapekit_patched", False):
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
    cls._tapekit_patched = True


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
