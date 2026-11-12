"""fittingkit: compound rigid bodies and a compression joint made up on a copper pipe, for a yam_arena task file.

Written for the compression-joint task in this folder, from the helpers of our earlier tasks. Importing it is cheap
(NumPy only): everything that needs Isaac Sim is imported lazily, inside hooks that run once the simulator has
started.

What it adds to a plain task file:

- :class:`Body`: one rigid body (or a fixed fixture, without a mass) made of several boxes and cylinders; a colour
  with a fourth (alpha) component below 1 is drawn see-through (glass).
- :class:`Slider` and :class:`Hinge`: prismatic and revolute joints between two bodies.
- :class:`CompressionJoint`: a copper pipe joined to a valve's upward-facing compression inlet (our own model: the
  rings on the pipe, the socket, the nut's thread, the spanner on the nut), the nut and olive posed once the nut is on
  the thread.
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
    "Shape", "box", "cyl", "ring", "prism12", "Body", "body", "BodyState", "CompressionJoint", "Hinge", "add_hinge",
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


class CompressionJoint:
    """A copper pipe joined to a valve's upward-facing compression inlet: our own model (the rings on the pipe, the
    socket, the nut's thread, the spanner on the nut).

    The joint's axis is vertical at world (``axis_x``, ``axis_y``). The valve's thread stub's top is at world height
    ``stub``; the pipe (body ``pipe``: origin at its bottom end, z up its axis) is home with its end ``socket`` below
    that, standing on the socket's stop. The olive and the nut (bodies ``olive``, ``nut``: origins at their middles, z
    up their bores) are dropped over the pipe's top and fall down it.

    - The pipe's end below the stub's top is held on the axis (the socket; its chamfered mouth guides it in).
    - Rings: a ring lowered over the pipe's top, coaxial within ``capture``, is guided on (its bore's chamfer); on the
      pipe it slides and turns physically. The olive comes to rest in the stub's cone (``cone`` below the stub's
      top). The nut comes to rest on the stub's thread, or on the olive if the olive is under it.
    - The thread: a nut resting on the stub is on the thread; from then on we pose it. Turned by ``theta``
      (clockwise seen from above, tightening, positive), it runs down ``pitch`` per turn. With the olive under it, it is
      snug after ``run_on`` turns, and ``turns_past()`` counts from there; past ``crush`` turns the olive and the pipe
      are crushed (latched). With no olive under it, it runs down its ``free_turns`` and stops, loose.
    - The spanner (body ``spanner``: origin at its jaw's middle, x along its handle, z up) lying level with its jaw on
      the nut's flats (within ``flat_tol`` of a flat, every 60 degrees) is on: its jaw is held on the nut's axis and the
      nut turns with it. Drawn out along its handle by ``release``, it comes off."""

    def __init__(self, pipe, olive, nut, spanner, axis_x, axis_y, stub, socket=12.0e-3, nut_len=12.0e-3,
                 olive_len=7.0e-3, cone=2.0e-3, capture=4.0e-3, pitch=1.814e-3, run_on=1.0, free_turns=2.2,
                 crush=1.5, flat_tol=12.0, release=4.0e-3, k_guide=200.0, k_hold=800.0, k_pipe=300.0, pipe_len=0.06):
        self.pipe, self.olive, self.nut, self.spanner = pipe, olive, nut, spanner
        self.pipe_len = float(pipe_len)
        self.axis = np.array([float(axis_x), float(axis_y)])
        self.stub, self.socket = float(stub), float(socket)
        self.nut_len, self.olive_len, self.cone = nut_len, olive_len, cone
        self.capture, self.pitch, self.run_on, self.free_turns, self.crush = capture, pitch, run_on, free_turns, crush
        self.flat_tol, self.release = math.radians(flat_tol), release
        self.k_guide, self.k_hold, self.k_pipe = k_guide, k_hold, k_pipe  # (light parts: soft enough to stay stable)
        self.dt = 1.0 / 240.0
        self.origin = np.zeros(3)
        self.poses = {}
        self.clear()

    @property
    def bodies(self) -> list[str]:
        return [self.pipe, self.olive, self.nut, self.spanner]

    def clear(self) -> None:
        self.on_pipe = {self.olive: False, self.nut: False}
        self.ring_z = {self.olive: 0.0, self.nut: 0.0}
        self.end_z = 1.0
        self.pipe_in = False
        self.threaded = False
        self.theta = 0.0
        self.olive_in = False
        self.crushed = False
        self.furthest = 0.0  # the most turns past snug reached (the olive's compression; backing off does not undo it)
        self.engaged = False
        self._psi = 0.0

    def home_z(self) -> float:
        return self.stub - self.socket

    def turns(self) -> float:
        return self.theta / (2 * math.pi)

    def turns_past(self) -> float:
        return max(0.0, self.turns() - self.run_on) if self.olive_in else 0.0

    def pipe_home(self, tol=1.0e-3) -> bool:
        return self.pipe_in and abs(self.end_z - self.home_z()) <= tol

    def sealed(self) -> bool:
        return (self.threaded and self.olive_in and not self.crushed and self.pipe_home()
                and 0.75 <= self.turns_past() <= 1.25)

    def update(self, frames: dict, acc: dict) -> None:
        fp, fo, fn, fs = (frames.get(n) for n in (self.pipe, self.olive, self.nut, self.spanner))
        if fp is None or fo is None or fn is None:
            return
        if not all(np.all(np.isfinite(f.pos)) for f in (fp, fo, fn)):
            return
        o = self.origin
        end = fp.pos - o
        self.end_z = float(end[2])
        lat = end[:2] - self.axis
        # the socket holds the pipe's end on the axis (its chamfered mouth guides it in)
        if self.end_z < self.stub + 4.0e-3 and float(np.hypot(*lat)) < 8.0e-3:
            k = self.k_pipe if self.end_z < self.stub else self.k_guide
            f = -k * lat - 1.0 * np.asarray(fp.vel[:2])
            _wrench(acc, self.pipe, fp, np.array([f[0], f[1], 0.0]), fp.pos)
            up = fp.rot[:, 2]
            if self.end_z < self.stub:  # and upright
                _torque(acc, self.pipe, 0.02 * np.cross(up, np.array([0.0, 0.0, 1.0])) - 3.0e-4 * np.asarray(fp.spin))
        self.pipe_in = self.end_z < self.stub and float(np.hypot(*lat)) < 2.0e-3
        top = self.end_z + float(np.dot(fp.rot[:, 2], [0, 0, 1])) * self.pipe_len
        # the rings
        for name, fr, length in ((self.olive, fo, self.olive_len), (self.nut, fn, self.nut_len)):
            p = fr.pos - o
            self.ring_z[name] = float(p[2])
            if name == self.nut and self.threaded:
                continue
            r = float(np.hypot(*(p[:2] - self.axis)))
            level = float(fr.rot[2, 2]) > math.cos(math.radians(15.0))
            if not self.on_pipe[name]:
                if r <= 1.5e-3 and p[2] - length / 2 < top - 2.0e-3 and level:
                    self.on_pipe[name] = True
                elif r <= self.capture and top - 2.0e-3 < p[2] - length / 2 < top + 0.012 and level:
                    f = -self.k_guide * (p[:2] - self.axis) - 0.5 * np.asarray(fr.vel[:2])  # its bore's chamfer
                    _wrench(acc, name, fr, np.array([f[0], f[1], 0.0]), fr.pos)
            elif p[2] - length / 2 > top + 1.0e-3:
                self.on_pipe[name] = False  # lifted off the top
            else:  # on the pipe it rides square and centred on it, so it slides down freely
                f = -self.k_guide * (p[:2] - self.axis) - 0.5 * np.asarray(fr.vel[:2])
                _wrench(acc, name, fr, np.array([f[0], f[1], 0.0]), fr.pos)
                up = fr.rot[:, 2]
                _torque(acc, name, 0.01 * np.cross(up, np.array([0.0, 0.0, 1.0])) - 1.0e-4 * np.asarray(fr.spin))
        # the olive rests in the stub's cone
        ob = self.ring_z[self.olive] - self.olive_len / 2
        if self.on_pipe[self.olive] and self.pipe_in and ob < self.stub - self.cone:
            f = min(5.0, 300.0 * (self.stub - self.cone - ob) - 0.5 * float(fo.vel[2]))
            _wrench(acc, self.olive, fo, np.array([0.0, 0.0, max(0.0, f)]), fo.pos)
        # the nut rests on the stub's thread (or on the olive), and is then on the thread
        if not self.threaded and self.on_pipe[self.nut] and self.pipe_in:
            nb = self.ring_z[self.nut] - self.nut_len / 2
            if nb < self.stub + 0.5e-3:
                self.threaded, self.theta = True, 0.0
                self.olive_in = (self.on_pipe[self.olive] and self.ring_z[self.olive] < self.ring_z[self.nut])
            elif nb < self.stub + 3.0e-3:  # settling onto the stub
                f = 300.0 * (self.stub + 0.5e-3 - nb) - 0.5 * float(fn.vel[2])
                if f > 0:
                    _wrench(acc, self.nut, fn, np.array([0.0, 0.0, min(5.0, f)]), fn.pos)
        # the spanner
        if fs is not None and self.threaded:
            jaw = fs.pos - o
            xl = fs.rot[:, 0]
            psi = math.atan2(xl[1], xl[0])  # its handle's heading (seen from above, anticlockwise +)
            nut_z = self._nut_z()
            d = jaw - np.array([*self.axis, nut_z])
            radial = float(np.hypot(d[0], d[1]))
            level = float(fs.rot[2, 2]) > math.cos(math.radians(12.0))
            if not self.engaged:
                flats = abs(((-psi - self.theta) + math.pi / 6) % (math.pi / 3) - math.pi / 6) <= self.flat_tol
                if radial < 3.0e-3 and abs(d[2]) < 5.0e-3 and level and flats:
                    self.engaged, self._psi = True, psi
            else:
                if float(np.dot(d, xl)) > self.release or abs(d[2]) > 8.0e-3:
                    self.engaged = False  # drawn off along its handle
                else:
                    perp = d - xl * max(0.0, float(np.dot(d, xl)))  # held on the axis, free to draw off outward
                    perp[2] = 0.0
                    f = -self.k_hold * perp - 2.0 * np.array([fs.vel[0], fs.vel[1], 0.0])
                    _wrench(acc, self.spanner, fs, f, fs.pos)
                    turn = -_wrap_angle(psi - self._psi)  # clockwise from above tightens
                    self._psi = psi
                    new = self.theta + turn
                    top_t = self.run_on + self.crush + 0.5 if self.olive_in else self.free_turns
                    held = min(max(new, -0.01), top_t * 2 * math.pi)
                    if held != new:  # against a stop: the nut holds the spanner
                        _torque(acc, self.spanner, np.array([0.0, 0.0, 2.0 * (new - held)]))
                    self.theta = held
                    self.furthest = max(self.furthest, self.turns_past())
                    if self.turns_past() > self.crush:
                        self.crushed = True
                    if self.theta < 0.0:
                        self.threaded = False
        self._draw()

    def _nut_z(self) -> float:
        """The threaded nut's middle's height."""
        base = self.stub - 1.0e-3 + self.nut_len / 2  # resting on the thread's start (or on the olive: the same height)
        e = self.pitch * self.turns()
        if self.olive_in:  # snug on the olive after ``run_on`` turns; then it compresses it slowly
            e = min(e, self.pitch * self.run_on) + 0.15 * max(0.0, e - self.pitch * self.run_on)
        return base - e

    def _draw(self) -> None:
        self.poses = {}
        if not self.threaded:
            return
        o = self.origin
        self.poses[self.nut] = (np.array([*self.axis, self._nut_z()]) + o, yaw_quat(-self.theta))
        if self.olive_in:
            z = self.stub - self.cone + self.olive_len / 2 - 0.15 * self.pitch * self.turns_past()
            self.poses[self.olive] = (np.array([*self.axis, z]) + o, yaw_quat(0.0))


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
    if getattr(anchor, "_fittingkit_hooked", False):
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
    anchor._fittingkit_hooked = True


def _patch_scene(scene) -> None:
    if getattr(scene, "_fittingkit_patched", False):
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
    scene._fittingkit_patched = True


def _patch_embodiment(ie) -> None:
    cls = ie.YamArenaEmbodiment
    if getattr(cls, "_fittingkit_patched", False):
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
    cls._fittingkit_patched = True


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
