import Video
import Setting
import mediapipe.python.solutions.hands as hands
import mediapipe.python.solutions.pose as pose
TopCam = Video.Cam(Setting.CamSetting.TopView)

Hand = hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.9)
Pose = pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.9)

while TopCam.cam_port.isOpened():
    TopCam.run_cam(Hand, Pose, 'TopCam')
TopCam.release()