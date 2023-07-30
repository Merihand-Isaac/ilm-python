import cv2
import Arm_Draw
import Arm_Connections


def cam_off():
    if cv2.waitKey(5) & 0xFF == 27:
        exit(0)


def cam_process(confidence_hand, confidence_pose, image):
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results_hand = confidence_hand.process(image)
    results_pose = confidence_pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return results_hand, results_pose, image


class Cam:
    def __init__(self, cam_port):
        self.cam_port = cv2.VideoCapture(cv2.CAP_DSHOW + cam_port)

    def run_cam(self, confidence_hand, confidence_pose, cam_name):
        success, image = self.cam_port.read()
        if not success:
            print("Cam is not working...")
            cam_off()
        process_result_hand, process_result_pose, image = cam_process(confidence_hand, confidence_pose, image)
        Arm_Draw.hand_draw_landmarks_connections(process_result_hand, image, Arm_Connections.NEW_HAND_CONNECTIONS)
        Arm_Draw.pose_draw_landmarks_connections(process_result_pose, image, Arm_Connections.NEW_POSE_CONNECTIONS)

        cv2.imshow('corp.MeriHand ' + cam_name + 'View - by Isaac', cv2.flip(image, 1))

        cam_off()
