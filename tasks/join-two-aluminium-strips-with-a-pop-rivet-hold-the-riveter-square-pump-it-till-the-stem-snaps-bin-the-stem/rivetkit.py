"""rivetkit: compound rigid bodies, sprung hinges and a blind rivet set by a hand riveter, for a yam_arena task file.

Written for the riveting task in this folder, from the helpers of our earlier tasks. Importing it is cheap
(NumPy only): everything that needs Isaac Sim is imported lazily, inside hooks that run once the simulator has
started.

What it adds to a plain task file:

- :class:`Body`: one rigid body (or a fixed fixture, without a mass) made of several boxes and cylinders; a colour
  with a fourth (alpha) component below 1 is drawn see-through (glass).
- :class:`Hinge`: a revolute joint between two bodies, with a spring, a brake and limits (a riveter's handle).
- :class:`Riveting`: a blind rivet loaded, inserted and set by a hand riveter (our own model), drawn by bodies we
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

__all__ = ["Shape", "box", "cyl", "ring", "prism12", "Body", "body", "BodyState", "Riveting", "Hinge", "add_hinge", "no_gravity",
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


class Riveting:
    """Blind rivets set by a hand riveter: our own model (PhysX cannot pull a rivet's stem or upset its tail).

    The riveter (body ``riveter``, its origin at the tip of its nose, its z axis up the nose) has a sprung handle (body
    ``handle``) hinged to it about the riveter's y axis; closing the handles turns it to smaller angles. Its nose does
    not collide: this model is its contact with what is under it. Rivets (``rivets``: name, head-top position in the
    holder) stand stem up in a holder; each is drawn by a body we pose (the head and shank), its stem by another
    (``stems``), the set tail by ``tail``.

    - Under the nose's tip, a surface pushes the riveter up (``k`` N/m on how far the tip is below it, damped): a
      rivet's head (loaded, or under the nose), its stem's tip (the nose on it but off its axis), the holder's or the
      work's top. So the riveter stands on what it is lowered onto. With a rivet's stem in the nosepiece and the rivet
      in the holder or in the hole, the nose cannot slide off it sideways (``k_side`` N/m towards the rivet's axis):
      pressed on it, the riveter can only tip about its nose.
    - Loading: the nose lowered over a rivet's stem, within ``load_tol`` of its axis (the nosepiece's countersunk
      mouth guides the stem in), until it sits on its head: the rivet is in the nose, and goes where the nose goes.
    - Inserting: a loaded rivet's shank goes into the work's hole only within ``hole_tol`` of its axis (otherwise the
      shank's end stands on the work); pushed in till its head is on the work, it is in the hole.
    - Pulling: each closing of the handles, with the rivet in the hole, pulls the stem ``pull_per_rad`` per radian
      closed (opening them pulls nothing, as the jaws slip back). Pulled with the riveter tilted over ``max_tilt``, off
      the head's axis by more than ``hole_tol`` + 0.5 mm, or lifted off it, the rivet is spoilt (set crooked or
      loose: latched). At ``set_pull`` the stem snaps: the rivet is set, the stem stays in the nose.
    - The handles opened past ``release_angle`` let the spent stem drop out of the nose: it falls straight down onto
      what is under it (the scrap tray, the work or the bench). Lifted off the head before any pull, the riveter takes
      the rivet out of the hole with it."""

    def __init__(self, riveter, handle, rivets, stems, tail, hole_xy, work_top, work_bottom, work_box, hole_tol,
                 holder_top, holder_box, tray_box, tray_floor, bench_top, hidden_z, head_t=1.5e-3, shank=12e-3, stem_len=25e-3,
                 load_tol=3.0e-3, k=2000.0, damping=3.0, k_side=3000.0, pull_per_rad=5.7e-3, set_pull=6e-3, max_tilt_deg=6.0,
                 release_angle=math.radians(23.0)):
        self.riveter, self.handle, self.tail = riveter, handle, tail
        self.rivets = [(n, np.asarray(p, float)) for n, p in rivets]
        self.stems = list(stems)
        self.hole = np.asarray(hole_xy, float)
        self.work_top, self.work_bottom, self.work_box = float(work_top), float(work_bottom), work_box
        self.hole_tol, self.holder_top = float(hole_tol), float(holder_top)
        self.holder_box, self.tray_box = holder_box, tray_box  # (x0, x1, y0, y1), env frame
        self.tray_floor, self.bench_top, self.hidden_z = float(tray_floor), float(bench_top), float(hidden_z)
        self.head_t, self.shank, self.stem_len, self.load_tol = head_t, shank, stem_len, load_tol
        self.k, self.damping, self.k_side = float(k), float(damping), float(k_side)
        self.pull_per_rad, self.set_pull = pull_per_rad, set_pull
        self.max_tilt = math.radians(max_tilt_deg)
        self.release_angle = release_angle
        self.dt = 1.0 / 240.0
        self.origin = np.zeros(3)
        self.poses = {}
        self.clear()

    @property
    def bodies(self) -> list[str]:
        return [self.riveter, self.handle, *[n for n, _ in self.rivets], *self.stems, self.tail]

    def clear(self) -> None:
        self.state = ["holder"] * len(self.rivets)  # holder, loaded, inserted, set, out (knocked off or dropped)
        self.active = None  # the rivet in the nose or the hole
        self.pull = 0.0
        self.spoilt = False
        self.spoil_reason = ""
        self.stem_in_nose = False
        self.stem_at = None  # where the spent stem lies (env frame), once dropped
        self.stem_place = ""  # "tray", "work" or "bench"
        self.angle = None
        self.tilt = 0.0
        self.peak_push = 0.0
        self._shown: dict = {}

    # -- helpers ---------------------------------------------------------------------------------------------------
    @staticmethod
    def _in(box, p) -> bool:
        return box[0] <= p[0] <= box[1] and box[2] <= p[1] <= box[3]

    def _handle_angle(self, fr, fh) -> float:
        """The sprung handle's angle from upright in the riveter's frame (rad): positive with its top away from the
        fixed handle (open), smaller as the handles close."""
        rel = fr.rot.T @ fh.rot  # the handle turned about the riveter's y axis
        return -math.atan2(rel[0, 2], rel[2, 2])

    def set_ok(self) -> bool:
        return bool(self.active is not None and self.state[self.active] == "set" and not self.spoilt)

    # -- the model -------------------------------------------------------------------------------------------------
    def update(self, frames: dict, acc: dict) -> None:
        fr, fh = frames.get(self.riveter), frames.get(self.handle)
        if fr is None or fh is None:
            return
        tip_w = fr.pos  # the nose's tip (the riveter's origin)
        tip = tip_w - self.origin
        up = fr.rot[:, 2]
        self.tilt = math.acos(max(-1.0, min(1.0, float(up[2]))))
        angle = self._handle_angle(fr, fh)
        closing = 0.0 if self.angle is None else max(0.0, self.angle - angle)
        self.angle = angle
        loaded = self.active is not None and self.state[self.active] == "loaded"
        inserted = self.active is not None and self.state[self.active] in ("inserted", "set")
        surface = None  # the height under the nose's tip
        off_hole = float(np.hypot(*(tip[:2] - self.hole)))
        if loaded:
            if off_hole <= self.hole_tol:  # over the hole: the shank goes in
                if tip[2] - self.head_t <= self.work_top + 1e-4:  # its head on the work: in the hole
                    self.state[self.active] = "inserted"
                    inserted, loaded = True, False
            else:  # the shank's end stands on what is under it
                fl = self._floor(tip)
                surface = None if fl is None else fl + self.head_t + self.shank
        if inserted:
            head_top = self.work_top + self.head_t
            if off_hole <= self.hole_tol + 0.5e-3:
                surface = head_top
            elif self._over_work(tip):
                surface = self.work_top
            on_head = off_hole <= self.hole_tol + 0.5e-3 and tip[2] <= head_top + 1e-3
            if self.state[self.active] == "inserted":
                if closing > 0.0:
                    if not on_head or self.tilt > self.max_tilt:
                        self.spoilt = True
                        self.spoil_reason = "crooked" if self.tilt > self.max_tilt else "loose"
                    self.pull += closing * self.pull_per_rad
                    if self.pull >= self.set_pull:
                        self.state[self.active] = "set"
                        self.stem_in_nose = True
                elif tip[2] > head_top + 2e-3 and self.pull == 0.0:  # lifted before any pull: it comes out
                    self.state[self.active] = "loaded"
                elif tip[2] > head_top + 1e-3 and self.pull > 0.0 and not self.spoilt:
                    self.spoilt, self.spoil_reason = True, "loose"  # the jaws had it: pulled out of true
        if not loaded and not inserted:
            # a rivet in the holder under the nose: load it, or stand on its stem
            for i, (name, head) in enumerate(self.rivets):
                if self.state[i] != "holder":
                    continue
                off = float(np.hypot(*(tip[:2] - head[:2])))
                if off <= self.load_tol and tip[2] <= head[2] + 1e-3 and self.active is None:
                    self.state[i], self.active = "loaded", i
                    surface = head[2]
                    break
                if off <= self.load_tol and tip[2] <= head[2] + self.stem_len:
                    surface = head[2]  # going down over the stem
                elif off <= self.load_tol + 2.5e-3 and tip[2] <= head[2] + self.stem_len:
                    surface = head[2] + self.stem_len  # the nose on the stem's tip, off its axis
            if surface is None:
                surface = self._floor(tip)
        elif surface is None:
            surface = self._floor(tip)
        if self.stem_in_nose and angle >= self.release_angle:
            self.stem_in_nose = False
            self.stem_place = ("tray" if self._in(self.tray_box, tip) else
                               "work" if self._over_work(tip) else "bench")
            z = (self.tray_floor if self.stem_place == "tray" else
                 self.work_top + self.head_t if self.stem_place == "work" else self.bench_top)
            self.stem_at = np.array([tip[0], tip[1], z])
        # the nose held on the rivet's axis: its stem in the nosepiece, the rivet in the holder or the hole
        axis_xy = None
        if inserted and off_hole <= self.hole_tol + 0.5e-3 and tip[2] <= self.work_top + self.head_t + 2e-3:
            axis_xy = self.hole
        elif (loaded and off_hole <= self.hole_tol
              and tip[2] - self.head_t - self.shank <= self.work_top + 1e-3):  # the shank's end in the hole's mouth
            axis_xy = self.hole
        else:
            for i, (name, head) in enumerate(self.rivets):
                if (self.state[i] in ("holder", "loaded") and np.hypot(*(tip[:2] - head[:2])) <= self.load_tol
                        and tip[2] <= head[2] + self.stem_len):
                    axis_xy = head[:2]
                    break
        if axis_xy is not None:
            d = np.asarray(axis_xy) - tip[:2]
            side = self.k_side * d - 2.0 * np.asarray(fr.vel[:2])
            _wrench(acc, self.riveter, fr, np.array([side[0], side[1], 0.0]), tip_w)
        # the riveter stands on the surface under its nose
        push = 0.0
        if surface is not None and tip[2] < surface:
            depth = surface - tip[2]
            push = max(0.0, self.k * depth - self.damping * float(fr.vel[2]))
            _wrench(acc, self.riveter, fr, np.array([0.0, 0.0, push]), tip_w)
        self.peak_push = max(self.peak_push, push)
        self._draw(fr, tip_w)

    def _over_work(self, tip) -> bool:
        return self._in(self.work_box, tip)

    def _floor(self, tip):
        """The height of what is under the nose apart from rivets: the work, the holder, the tray or None."""
        if self._over_work(tip):
            return self.work_top
        if self._in(self.holder_box, tip):
            return self.holder_top
        return None

    # -- drawing ---------------------------------------------------------------------------------------------------
    def _pose(self, name, pos, quat=(0.0, 0.0, 0.0, 1.0)) -> None:
        key = tuple(np.round(np.r_[pos, quat], 5))
        if self._shown.get(name) != key:
            self._shown[name] = key
            self.poses[name] = (np.asarray(pos, float), tuple(quat))

    def _draw(self, fr, tip_w) -> None:
        self.poses = {}
        q_nose = matrix_to_quat_rel(fr.rot)
        hidden = np.array([self.hole[0], self.hole[1], self.hidden_z]) + self.origin
        for i, (name, head) in enumerate(self.rivets):
            st = self.state[i]
            stem = self.stems[i]
            if st == "holder":
                self._pose(name, head + self.origin)
                self._pose(stem, head + self.origin)
            elif st == "loaded":
                self._pose(name, tip_w, q_nose)
                self._pose(stem, tip_w, q_nose)
            elif st in ("inserted", "set"):
                h = np.array([self.hole[0], self.hole[1], self.work_top + self.head_t]) + self.origin
                self._pose(name, h)
                if st == "inserted":
                    self._pose(stem, h + np.array([0.0, 0.0, self.pull]))
                elif self.stem_in_nose:
                    self._pose(stem, tip_w + fr.rot @ np.array([0.0, 0.0, self.set_pull]), q_nose)
                elif self.stem_at is not None:  # lying where it fell, along x
                    self._pose(stem, self.stem_at + self.origin + np.array([0.0, 0.0, 0.0012]),
                               (0.0, math.sin(math.pi / 4), 0.0, math.cos(math.pi / 4)))
        if self.set_ok() or (self.active is not None and self.state[self.active] == "set"):
            self._pose(self.tail, np.array([self.hole[0], self.hole[1], self.work_bottom]) + self.origin)
        else:
            self._pose(self.tail, hidden)


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
    if getattr(anchor, "_rivetkit_hooked", False):
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
    anchor._rivetkit_hooked = True


def _patch_scene(scene) -> None:
    if getattr(scene, "_rivetkit_patched", False):
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
    scene._rivetkit_patched = True


def _patch_embodiment(ie) -> None:
    cls = ie.YamArenaEmbodiment
    if getattr(cls, "_rivetkit_patched", False):
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
    cls._rivetkit_patched = True


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
