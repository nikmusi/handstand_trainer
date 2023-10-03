import numpy as np

def calculate_joint_angle(a, b, c):
    """Calculates the angle of the joint given as "b" inside a plane. If z-y-koordinates are given, the angle inside
    the z-y-plane will be calculated. The maximal joint angle is 180°

    Args:
        a (list): (x,y)-Koordinates of first outer joint
        b (list): (x,y)-Koordinates of main joint (for angle calculation)
        c (list): (x,y)-Koordinates of second outer joint

    Returns:
        The Angle (float) between the vectors a->b and b->c
    """
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180 / np.pi)

    if angle > 180:
        angle = 360 - angle

    return angle