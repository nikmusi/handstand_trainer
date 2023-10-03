import cv2
import mediapipe as mp

from handstand_trainer import HandstandTrainer


def run(path):
    """Runs the source video and analyses the handstand pose.
    Args:
        video_path (str): path to the source video, loads webcam if string is empty
    """

    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose         # use pose estimation model

    # get the video stream
    if path:
        # load the source video
        cap = cv2.VideoCapture(path)

        # specify_output
        out = cv2.VideoWriter('results/output.avi', cv2.VideoWriter_fourcc(*'MJPG'), 20.0, (1080, 1920))
    else:
        # if the path is empty - play from the camera
        cap = cv2.VideoCapture(0)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:

        # get the video feed for pose estimation (from webcam)
        while cap.isOpened():
            # read the current frame
            ret, frame = cap.read()

            # Change image format to rgb
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # use pose estimation on the current image
            results = pose.process(image)

            # change colors to bgr
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # analyse the handstand position
            try:
                trainer = HandstandTrainer(results.pose_landmarks.landmark, mp_pose.PoseLandmark, image)
                image = trainer.analyse_handstand_pose()
            except:
                pass

            # draw the skeleton on the image
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

            # show the current image
            cv2.imshow('Video feed', image)
            try:
                # write the results video - if source is also a video
                out.write(image)
            except:
                pass

            # close the video feed
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    # clean up
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    video_path = r"video_input/test.mp4"
    run(video_path)
