import cv2

from utils.trigonometry import calculate_joint_angle as angle


class HandstandTrainer:
    """Analyse Handstands by three very basic aspects, to achieve handstand with minimal strength requirements:
            1. distance of hands
            2. symmetrical weight distribution on both arms
            3. straight arms
    It is assumed that the handstand is performed with the back facing the camera, when in position.
    """

    def __init__(self, pose_landmarks, landmark_map, image):
        # constructor
        self.pose = pose_landmarks
        self.map = landmark_map
        self.image = image

        # init results values
        self.hand_distance = None
        self.symmetry = None
        self.straight_arms = None

    def analyse_handstand_pose(self):
        """Analysing every four aspects of the handstand, using the corresponding methods for every aspect"""

        # run all four aspect analyses
        self._analyse_hand_distance_and_symmetry()
        self._analyse_straight_arms()

        # print results on the image
        self._print_results_on_image()

        return self.image

    def _analyse_hand_distance_and_symmetry(self):
        """Analysing the hand distance for minimal strength requirement. The best assumed hand position is 90° regarding
        the x-y-plane
        """
        # get the relevant joints
        right_shoulder = [self.pose[self.map.RIGHT_SHOULDER.value].x, self.pose[self.map.RIGHT_SHOULDER.value].y]
        right_wrist = [self.pose[self.map.RIGHT_WRIST.value].x, self.pose[self.map.RIGHT_WRIST.value].y]
        right_ground = [self.pose[self.map.RIGHT_WRIST.value].x + 1, self.pose[self.map.RIGHT_WRIST.value].y]

        left_shoulder = [self.pose[self.map.LEFT_SHOULDER.value].x, self.pose[self.map.LEFT_SHOULDER.value].y]
        left_wrist = [self.pose[self.map.LEFT_WRIST.value].x, self.pose[self.map.LEFT_WRIST.value].y]
        left_ground = [self.pose[self.map.LEFT_WRIST.value].x - 1, self.pose[self.map.LEFT_WRIST.value].y]

        # calculate the angle, as main joint the wrist is used - outer joints are elbow and the ground plane
        right_wrist_angle = angle(right_shoulder, right_wrist, right_ground)
        left_wrist_angle = angle(left_shoulder, left_wrist, left_ground)

        # ---- DISTANCE classification----
        if right_wrist_angle < 60 and left_wrist_angle < 60:
            # case 1: hands to broad apart (to narrow stance is neglected for simplicity - very uncommon error)
            self.hand_distance = "Hand Position: Narrow your hand position significantly!"

        elif right_wrist_angle < 70 and left_wrist_angle < 70:
            # case 2: hands to broad apart (to narrow stance is neglected for simplicity - very uncommon error)
            self.hand_distance = "Hand Position: Narrow your hand position slightly!"

        else:
            # otherwise it's fine
            self.hand_distance = "Hand Position: Well Done!"

        # ---- SYMMETRY classification----
        if abs(right_wrist_angle - left_wrist_angle) >= 5:
            # case 1: wrists not similar angles
            self.symmetry = "Symmetry: Non-symmetric weight distribution"

        else:
            # otherwise it's fine -> just a simple project
            self.symmetry = "Symmetry: Well Done!"

    def _analyse_straight_arms(self):
        """Analyses, if the arms are straight"""
        # get the relevant joints
        right_shoulder = [self.pose[self.map.RIGHT_SHOULDER.value].x, self.pose[self.map.RIGHT_SHOULDER.value].y]
        right_elbow = [self.pose[self.map.RIGHT_ELBOW.value].x, self.pose[self.map.RIGHT_ELBOW.value].y]
        right_wrist = [self.pose[self.map.RIGHT_WRIST.value].x, self.pose[self.map.RIGHT_WRIST.value].y]

        left_shoulder = [self.pose[self.map.LEFT_SHOULDER.value].x, self.pose[self.map.LEFT_SHOULDER.value].y]
        left_elbow = [self.pose[self.map.LEFT_ELBOW.value].x, self.pose[self.map.LEFT_ELBOW.value].y]
        left_wrist = [self.pose[self.map.LEFT_WRIST.value].x, self.pose[self.map.LEFT_WRIST.value].y]

        # calculate joint angles
        right_elbow_angle = angle(right_shoulder, right_elbow, right_wrist)
        left_elbow_angle = angle(left_shoulder, left_elbow, left_wrist)

        # ---- STRAIGHT ARMS classification ----
        if right_elbow_angle < 170 and left_elbow_angle < 170:
            # case 1: arms are band significantly
            self.straight_arms = "Arms: Straighten your arms!"

        else:
            # otherwise it's fine
            self.straight_arms = "Arms: Well done!"

    def _print_results_on_image(self):
        """Prints all calculated results onto the image """
        # general setup
        org = [10, 30]
        dist = 35
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (255, 0, 0)
        thickness = 1
        count = 0

        # hand distance
        for aspect in [self.hand_distance, self.symmetry, self.straight_arms]:
            if aspect is not None:

                cv2.putText(self.image, aspect, (org[0], org[1]+count*dist), font, font_scale, color, thickness,
                            cv2.LINE_AA)
                count += 1
