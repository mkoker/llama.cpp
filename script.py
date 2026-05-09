from manim import *
import math
import numpy as np


BG = "#101218"
SKIN = "#D89B72"
SHIRT = "#1E88E5"
SHORTS = "#263238"
SHOE = "#111111"
OUTLINE = "#071019"


class RunningMan(VGroup):
    """Filled, proportioned runner assembled from vector body segments.

    Height is normalized to ``height``.  The design uses an athletic 7.5-head
    figure: head ~= 13% of body height, torso ~= 30%, legs ~= 42%, arms ~= 34%.
    Limb segments are rounded capsules instead of stick lines so later animation
    can rotate them while preserving a solid silhouette.
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

        self.pelvis = ORIGIN + DOWN * 0.22 * height
        self.neck = self.pelvis + UP * self.torso_h
        self.head_center = self.neck + UP * (self.head_r * 1.45)
        self.left_shoulder = self.neck + LEFT * (self.shoulder_w / 2)
        self.right_shoulder = self.neck + RIGHT * (self.shoulder_w / 2)
        self.left_hip = self.pelvis + LEFT * (self.hip_w / 2)
        self.right_hip = self.pelvis + RIGHT * (self.hip_w / 2)

        self.torso = Polygon(
            self.left_shoulder,
            self.right_shoulder,
            self.right_hip + RIGHT * 0.03 * height,
            self.left_hip + LEFT * 0.03 * height,
        ).set_fill(SHIRT, opacity=1).set_stroke(OUTLINE, width=2)
        self.chest = Ellipse(width=self.shoulder_w * 1.03, height=self.torso_h * 0.72)
        self.chest.move_to(self.neck + DOWN * (self.torso_h * 0.34))
        self.chest.set_fill(SHIRT, opacity=1).set_stroke(OUTLINE, width=2)

        self.head = Circle(radius=self.head_r).move_to(self.head_center)
        self.head.set_fill(SKIN, opacity=1).set_stroke(OUTLINE, width=2)
        self.neck_shape = RoundedRectangle(
            width=self.head_r * 0.62,
            height=self.head_r * 0.70,
            corner_radius=self.head_r * 0.18,
        ).move_to(self.neck + UP * self.head_r * 0.20)
        self.neck_shape.set_fill(SKIN, opacity=1).set_stroke(OUTLINE, width=1.5)

        # A readable mid-stride pose.  Joints are kept as attributes for future
        # cycle animation while the rendered parts are filled capsules.
        self.left_elbow = self.left_shoulder + LEFT * 0.10 * height + DOWN * 0.15 * height
        self.left_hand = self.left_elbow + RIGHT * 0.12 * height + DOWN * 0.15 * height
        self.right_elbow = self.right_shoulder + RIGHT * 0.16 * height + UP * 0.04 * height
        self.right_hand = self.right_elbow + LEFT * 0.09 * height + DOWN * 0.15 * height
        self.left_knee = self.left_hip + LEFT * 0.12 * height + DOWN * 0.21 * height
        self.left_foot = self.left_knee + RIGHT * 0.18 * height + DOWN * 0.20 * height
        self.right_knee = self.right_hip + RIGHT * 0.15 * height + DOWN * 0.12 * height
        self.right_foot = self.right_knee + RIGHT * 0.10 * height + DOWN * 0.25 * height

        self.left_upper_arm = self._capsule(self.left_shoulder, self.left_elbow, self.arm_w, SKIN)
        self.left_forearm = self._capsule(self.left_elbow, self.left_hand, self.arm_w * 0.90, SKIN)
        self.right_upper_arm = self._capsule(self.right_shoulder, self.right_elbow, self.arm_w, SKIN)
        self.right_forearm = self._capsule(self.right_elbow, self.right_hand, self.arm_w * 0.90, SKIN)
        self.left_fist = self._joint_circle(self.left_hand, self.arm_w * 0.68, SKIN)
        self.right_fist = self._joint_circle(self.right_hand, self.arm_w * 0.68, SKIN)
        self.left_thigh = self._capsule(self.left_hip, self.left_knee, self.leg_w, SHORTS)
        self.left_shin = self._capsule(self.left_knee, self.left_foot, self.leg_w * 0.88, SKIN)
        self.right_thigh = self._capsule(self.right_hip, self.right_knee, self.leg_w, SHORTS)
        self.right_shin = self._capsule(self.right_knee, self.right_foot, self.leg_w * 0.88, SKIN)
        self.left_kneecap = self._joint_circle(self.left_knee, self.leg_w * 0.55, SHORTS)
        self.right_kneecap = self._joint_circle(self.right_knee, self.leg_w * 0.55, SHORTS)
        self.left_shoe = self._shoe(self.left_foot, facing=RIGHT)
        self.right_shoe = self._shoe(self.right_foot, facing=RIGHT)

        # Back limbs first, torso/head in front, front limbs last.
        self.add(
            self.right_upper_arm, self.right_forearm, self.right_fist,
            self.right_thigh, self.right_shin, self.right_kneecap, self.right_shoe,
            self.neck_shape, self.torso, self.chest, self.head,
            self.left_thigh, self.left_shin, self.left_kneecap, self.left_shoe,
            self.left_upper_arm, self.left_forearm, self.left_fist,
        )
        self.move_to(ORIGIN)

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

    def _shoe(self, foot, facing=RIGHT):
        shoe = RoundedRectangle(
            width=0.19 * self.body_height,
            height=0.045 * self.body_height,
            corner_radius=0.020 * self.body_height,
        )
        shoe.set_fill(SHOE, opacity=1).set_stroke(OUTLINE, width=1.2)
        shoe.move_to(foot + facing * 0.045 * self.body_height + DOWN * 0.012 * self.body_height)
        shoe.rotate(-0.10)
        return shoe


class RunningScene(Scene):
    def construct(self):
        self.camera.background_color = BG
        runner = RunningMan(height=3.4).move_to(ORIGIN)
        title = Text("Running Man Animation", font_size=34, color=WHITE).to_edge(UP, buff=0.5)
        self.add(title, runner)
        self.wait(1)
