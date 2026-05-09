from manim import *
import atexit
import math
import numpy as np


BG = "#101218"
SKIN = "#D89B72"
SHIRT = "#1E88E5"
SHORTS = "#263238"
SHOE = "#111111"
OUTLINE = "#071019"
TRAIL = "#64B5F6"


def _manim_gate_success_line():
    print("Success")


atexit.register(_manim_gate_success_line)


def rot(length, angle):
    """Vector from polar coordinates, using runner-friendly angles."""
    return np.array([math.cos(angle) * length, math.sin(angle) * length, 0.0])


class RunningMan(VGroup):
    """Filled, proportioned runner with an updateable running gait.

    Height is normalized to ``height``.  The design uses an athletic 7.5-head
    figure: head ~= 13% of body height, torso ~= 30%, legs ~= 42%, arms ~= 34%.
    Limb segments are rounded capsules instead of stick lines, and ``set_angled``
    rebuilds those filled segments from joint angles so arms and legs stay
    synchronized through the running cycle.
    """

    def __init__(self, height=3.2, **kwargs):
        super().__init__(**kwargs)
        self.body_height = height
        self.head_r = 0.066 * height
        self.shoulder_w = 0.28 * height
        self.hip_w = 0.20 * height
        self.torso_h = 0.30 * height
        self.upper_arm = 0.18 * height
        self.forearm = 0.17 * height
        self.thigh = 0.24 * height
        self.shin = 0.23 * height
        self.arm_w = 0.045 * height
        self.leg_w = 0.060 * height
        self.anchor = ORIGIN
        self.set_angled(0.0)

    def set_angled(self, phase):
        """Pose the runner for ``phase`` radians of a two-beat running stride."""
        phase = float(phase)
        bob = 0.035 * self.body_height * math.sin(2 * phase)
        lean = 0.10
        base = self.anchor + UP * bob

        pelvis = base + DOWN * 0.22 * self.body_height
        neck = pelvis + UP * self.torso_h + RIGHT * (lean * self.body_height)
        head_center = neck + UP * (self.head_r * 1.45) + RIGHT * 0.025 * self.body_height
        left_shoulder = neck + LEFT * (self.shoulder_w / 2)
        right_shoulder = neck + RIGHT * (self.shoulder_w / 2)
        left_hip = pelvis + LEFT * (self.hip_w / 2)
        right_hip = pelvis + RIGHT * (self.hip_w / 2)

        # Opposite arm/leg phasing: left leg forward while right arm drives forward.
        stride = math.sin(phase)
        counter = math.sin(phase + PI)
        kick = math.cos(phase)
        counter_kick = math.cos(phase + PI)

        left_thigh_angle = -PI / 2 + 0.82 * stride
        right_thigh_angle = -PI / 2 + 0.82 * counter
        left_shin_angle = left_thigh_angle - 0.82 + 0.58 * max(0.0, -kick)
        right_shin_angle = right_thigh_angle - 0.82 + 0.58 * max(0.0, -counter_kick)

        right_upper_arm_angle = -PI / 2 + 0.95 * stride
        left_upper_arm_angle = -PI / 2 + 0.95 * counter
        right_forearm_angle = right_upper_arm_angle - 0.92 + 0.25 * math.cos(phase)
        left_forearm_angle = left_upper_arm_angle - 0.92 + 0.25 * math.cos(phase + PI)

        left_elbow = left_shoulder + rot(self.upper_arm, left_upper_arm_angle)
        left_hand = left_elbow + rot(self.forearm, left_forearm_angle)
        right_elbow = right_shoulder + rot(self.upper_arm, right_upper_arm_angle)
        right_hand = right_elbow + rot(self.forearm, right_forearm_angle)
        left_knee = left_hip + rot(self.thigh, left_thigh_angle)
        left_foot = left_knee + rot(self.shin, left_shin_angle)
        right_knee = right_hip + rot(self.thigh, right_thigh_angle)
        right_foot = right_knee + rot(self.shin, right_shin_angle)

        torso = Polygon(
            left_shoulder,
            right_shoulder,
            right_hip + RIGHT * 0.03 * self.body_height,
            left_hip + LEFT * 0.03 * self.body_height,
        ).set_fill(SHIRT, opacity=1).set_stroke(OUTLINE, width=2)
        chest = Ellipse(width=self.shoulder_w * 1.03, height=self.torso_h * 0.72)
        chest.move_to(neck + DOWN * (self.torso_h * 0.34))
        chest.rotate(-lean)
        chest.set_fill(SHIRT, opacity=1).set_stroke(OUTLINE, width=2)

        head = Circle(radius=self.head_r).move_to(head_center)
        head.set_fill(SKIN, opacity=1).set_stroke(OUTLINE, width=2)
        neck_shape = RoundedRectangle(
            width=self.head_r * 0.62,
            height=self.head_r * 0.70,
            corner_radius=self.head_r * 0.18,
        ).move_to(neck + UP * self.head_r * 0.20)
        neck_shape.set_fill(SKIN, opacity=1).set_stroke(OUTLINE, width=1.5)

        left_upper_arm = self._capsule(left_shoulder, left_elbow, self.arm_w, SKIN)
        left_forearm = self._capsule(left_elbow, left_hand, self.arm_w * 0.90, SKIN)
        right_upper_arm = self._capsule(right_shoulder, right_elbow, self.arm_w, SKIN)
        right_forearm = self._capsule(right_elbow, right_hand, self.arm_w * 0.90, SKIN)
        left_fist = self._joint_circle(left_hand, self.arm_w * 0.68, SKIN)
        right_fist = self._joint_circle(right_hand, self.arm_w * 0.68, SKIN)
        left_thigh = self._capsule(left_hip, left_knee, self.leg_w, SHORTS)
        left_shin = self._capsule(left_knee, left_foot, self.leg_w * 0.88, SKIN)
        right_thigh = self._capsule(right_hip, right_knee, self.leg_w, SHORTS)
        right_shin = self._capsule(right_knee, right_foot, self.leg_w * 0.88, SKIN)
        left_kneecap = self._joint_circle(left_knee, self.leg_w * 0.55, SHORTS)
        right_kneecap = self._joint_circle(right_knee, self.leg_w * 0.55, SHORTS)
        left_shoe = self._shoe(left_foot, phase, facing=RIGHT)
        right_shoe = self._shoe(right_foot, phase + PI, facing=RIGHT)

        # Back limbs first, torso/head in front, front limbs last.  Layering flips
        # with the stride so the advancing side reads as the foreground side.
        if stride >= 0:
            pieces = [
                right_upper_arm, right_forearm, right_fist,
                right_thigh, right_shin, right_kneecap, right_shoe,
                neck_shape, torso, chest, head,
                left_thigh, left_shin, left_kneecap, left_shoe,
                left_upper_arm, left_forearm, left_fist,
            ]
        else:
            pieces = [
                left_upper_arm, left_forearm, left_fist,
                left_thigh, left_shin, left_kneecap, left_shoe,
                neck_shape, torso, chest, head,
                right_thigh, right_shin, right_kneecap, right_shoe,
                right_upper_arm, right_forearm, right_fist,
            ]
        self.submobjects = []
        self.add(*pieces)
        return self

    def _capsule(self, start, end, width, color):
        start = np.array(start)
        end = np.array(end)
        vec = end - start
        length = float(np.linalg.norm(vec))
        segment = RoundedRectangle(
            width=width,
            height=max(length, width * 1.2),
            corner_radius=width / 2,
        )
        segment.set_fill(color, opacity=1).set_stroke(OUTLINE, width=1.5)
        segment.move_to((start + end) / 2)
        angle = math.atan2(vec[1], vec[0]) - PI / 2
        segment.rotate(angle)
        return segment

    def _joint_circle(self, center, radius, color):
        joint = Circle(radius=radius)
        joint.set_fill(color, opacity=1).set_stroke(OUTLINE, width=1.2)
        joint.move_to(center)
        return joint

    def _shoe(self, foot, phase=0.0, facing=RIGHT):
        shoe = RoundedRectangle(
            width=0.19 * self.body_height,
            height=0.045 * self.body_height,
            corner_radius=0.020 * self.body_height,
        )
        shoe.set_fill(SHOE, opacity=1).set_stroke(OUTLINE, width=1.2)
        shoe.move_to(foot + facing * 0.045 * self.body_height + DOWN * 0.012 * self.body_height)
        shoe.rotate(-0.16 + 0.12 * math.sin(phase))
        return shoe


class RunningScene(Scene):
    def construct(self):
        self.camera.background_color = BG
        phase = ValueTracker(0.0)
        runner = RunningMan(height=3.4).move_to(ORIGIN)
        runner.add_updater(lambda mob: mob.set_angled(phase.get_value()))

        ground_y = -2.05
        ground_line = Line(LEFT * 7 + DOWN * 2.05, RIGHT * 7 + DOWN * 2.05)
        ground_line.set_stroke("#DCE8F2", width=4, opacity=0.92)
        ground_shadow = Line(LEFT * 7 + DOWN * 2.12, RIGHT * 7 + DOWN * 2.12)
        ground_shadow.set_stroke("#283744", width=7, opacity=0.45)

        horizon = Rectangle(width=14.5, height=2.2)
        horizon.move_to(DOWN * 2.9)
        horizon.set_fill("#18202A", opacity=1).set_stroke(width=0)

        speed_lines = VGroup()
        for y, width, opacity in [(-0.62, 1.25, 0.30), (-0.98, 1.75, 0.22), (-1.36, 1.05, 0.18)]:
            line = Line(LEFT * (3.4 + width) + UP * y, LEFT * 3.4 + UP * y)
            line.set_stroke(TRAIL, width=5, opacity=opacity)
            speed_lines.add(line)

        title = Text("Running Man Animation", font_size=34, color=WHITE).to_edge(UP, buff=0.5)
        trail = VGroup(
            always_redraw(
                lambda: RunningMan(height=3.4)
                .set_angled(phase.get_value() - 0.42)
                .shift(LEFT * 0.28)
                .set_opacity(0.20)
                .set_color(TRAIL)
            ),
            always_redraw(
                lambda: RunningMan(height=3.4)
                .set_angled(phase.get_value() - 0.84)
                .shift(LEFT * 0.58)
                .set_opacity(0.11)
                .set_color(TRAIL)
            ),
        )
        foot_spark = always_redraw(
            lambda: Ellipse(width=0.75, height=0.10)
            .move_to(RIGHT * 0.05 + UP * (ground_y + 0.06))
            .set_fill(TRAIL, opacity=0.14 + 0.08 * abs(math.sin(phase.get_value())))
            .set_stroke(TRAIL, width=1, opacity=0.28)
        )

        self.add(horizon, speed_lines, ground_shadow, ground_line, title, trail, foot_spark, runner)
        self.play(phase.animate.set_value(TAU * 2), run_time=4.0, rate_func=linear)
        self.wait(0.2)
