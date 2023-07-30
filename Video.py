import cv2
import mediapipe.python.solutions.holistic as holistic
import Arm_Draw as AD
import Arm_Connections as AC

def Camoff() :
    if cv2.waitKey(5) & 0xFF == 27:
        exit(0)

def CamProcess(Arm, image) :
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = Arm.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow('corp.MeriHand ver.Prototype - by Isaac', cv2.flip(image, 1))

    return results

def Cam_start() :
    Cam = cv2.VideoCapture(0)
    with holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as Arm:
        while Cam.isOpened():
            success, image = Cam.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            results = CamProcess(Arm, image)
            AD.Arm_draw_landmarks_connections(image, results.left_hand_landmarks, results.right_hand_landmarks, results.pose_landmarks)
            cv2.imshow('corp.MeriHand ver.Prototype - by Isaac', cv2.flip(image, 1))
            Camoff()
    Cam.release()