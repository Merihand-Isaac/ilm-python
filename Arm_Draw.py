import cv2
import numpy as np
import Arm_Connections as AC
import mediapipe.python.solutions.drawing_styles as ds
from typing import List, Mapping, Optional, Tuple, Union
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions.drawing_utils import DrawingSpec, RED_COLOR, _BGR_CHANNELS, _normalized_to_pixel_coordinates, _PRESENCE_THRESHOLD, _VISIBILITY_THRESHOLD, WHITE_COLOR

def pose_draw_landmarks_connections(
        image: np.ndarray,
        landmark_list: landmark_pb2.NormalizedLandmarkList,
        connections: Optional[List[Tuple[int, int]]] = None,
        landmark_drawing_spec: Union[DrawingSpec, Mapping[int, DrawingSpec]] = DrawingSpec(color = RED_COLOR),
        connection_drawing_spec: Union[DrawingSpec, Mapping[Tuple[int, int], DrawingSpec]] = DrawingSpec()):

    if not landmark_list:
        return

    image_rows, image_cols, _ = image.shape
    idx_to_coordinates = {}

    if image.shape[2] != _BGR_CHANNELS:
        raise ValueError('Input image must contain three channel bgr data.')

    for idx, landmark in enumerate(landmark_list.landmark):
        if idx in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]:
            continue
        if ((landmark.HasField('visibility') and landmark.visibility < _VISIBILITY_THRESHOLD) or (landmark.HasField('presence') and  landmark.presence < _PRESENCE_THRESHOLD)):
            continue
        landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y, image_cols, image_rows)
        if landmark_px:
            idx_to_coordinates[idx] = landmark_px

    if connections:
        num_landmarks = len(landmark_list.landmark)
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                raise ValueError(f'Landmark index is out of range. Invalid connection 'f'from landmark #{start_idx} to landmark #{end_idx}.')
            if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
                drawing_spec = connection_drawing_spec[connection] if isinstance(connection_drawing_spec, Mapping) else connection_drawing_spec
                cv2.line(image, idx_to_coordinates[start_idx], idx_to_coordinates[end_idx], drawing_spec.color, drawing_spec.thickness)

    if landmark_drawing_spec:
        for idx, landmark_px in idx_to_coordinates.items():
            drawing_spec = landmark_drawing_spec[idx] if isinstance(landmark_drawing_spec, Mapping) else landmark_drawing_spec
            circle_border_radius = max(drawing_spec.circle_radius + 1, int(drawing_spec.circle_radius * 1.2))
            cv2.circle(image, landmark_px, circle_border_radius, WHITE_COLOR, drawing_spec.thickness)
            cv2.circle(image, landmark_px, drawing_spec.circle_radius, drawing_spec.color, drawing_spec.thickness)


def hand_draw_landmarks_connections(
        image: np.ndarray,
        landmark_list: landmark_pb2.NormalizedLandmarkList,
        connections: Optional[List[Tuple[int, int]]] = None,
        landmark_drawing_spec: Union[DrawingSpec, Mapping[int, DrawingSpec]] = DrawingSpec(color=RED_COLOR),
        connection_drawing_spec: Union[DrawingSpec, Mapping[Tuple[int, int], DrawingSpec]] = DrawingSpec()):

    if not landmark_list:
        return

    image_rows, image_cols, _ = image.shape
    idx_to_coordinates = {}

    for idx, landmark in enumerate(landmark_list.landmark):
        if ((landmark.HasField('visibility') and landmark.visibility < _VISIBILITY_THRESHOLD) or (landmark.HasField('presence') and  landmark.presence < _PRESENCE_THRESHOLD)):
            continue
        landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y, image_cols, image_rows)
        if landmark_px:
            idx_to_coordinates[idx] = landmark_px

    if connections:
        num_landmarks = len(landmark_list.landmark)
        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                raise ValueError(
                    f'Landmark index is out of range. Invalid connection 'f'from landmark #{start_idx} to landmark #{end_idx}.')
            if start_idx in idx_to_coordinates and end_idx in idx_to_coordinates:
                drawing_spec = connection_drawing_spec[connection] if isinstance(connection_drawing_spec, Mapping) else connection_drawing_spec
                cv2.line(image, idx_to_coordinates[start_idx], idx_to_coordinates[end_idx], drawing_spec.color, drawing_spec.thickness)

    if landmark_drawing_spec:
        for idx, landmark_px in idx_to_coordinates.items():
            drawing_spec = landmark_drawing_spec[idx] if isinstance(landmark_drawing_spec, Mapping) else landmark_drawing_spec
            circle_border_radius = max(drawing_spec.circle_radius + 1, int(drawing_spec.circle_radius * 1.2))
            cv2.circle(image, landmark_px, circle_border_radius, WHITE_COLOR, drawing_spec.thickness)
            cv2.circle(image, landmark_px, drawing_spec.circle_radius, drawing_spec.color, drawing_spec.thickness)

## Pose - Left Hand Connect
def pose_to_Lhand_draw_connections(image: np.ndarray,
        Lhand_landmark_list: landmark_pb2.NormalizedLandmarkList,
        pose_landmark_list: landmark_pb2.NormalizedLandmarkList,
        connections: Optional[List[Tuple[int, int]]] = None,
        connection_drawing_spec: Union[DrawingSpec, Mapping[Tuple[int, int], DrawingSpec]] = DrawingSpec()):

    if not Lhand_landmark_list and pose_landmark_list:
        return

    image_rows, image_cols, _ = image.shape
    Left_idx_to_coordinates = {}


    for idx, landmark in enumerate(Lhand_landmark_list.landmark):
        if ((landmark.HasField('visibility') and landmark.visibility < _VISIBILITY_THRESHOLD) or (landmark.HasField('presence') and landmark.presence < _PRESENCE_THRESHOLD)):
            continue
        landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y, image_cols, image_rows)
        if landmark_px and idx == 0:
            Left_idx_to_coordinates[0] = landmark_px
            continue

    for idx, landmark in enumerate(pose_landmark_list.landmark):
        if ((landmark.HasField('visibility') and landmark.visibility < _VISIBILITY_THRESHOLD) or (landmark.HasField('presence') and  landmark.presence < _PRESENCE_THRESHOLD)):
            continue
        landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y, image_cols, image_rows)
        if landmark_px and idx == 13:
            Left_idx_to_coordinates[1] = landmark_px

    if connections:
        num_landmarks =len(Lhand_landmark_list.landmark) + len(pose_landmark_list.landmark)

        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                raise ValueError(f'Landmark index is out of range. Invalid connection 'f'from landmark #{start_idx} to landmark #{end_idx}.')
            if start_idx in Left_idx_to_coordinates and end_idx in Left_idx_to_coordinates:
                drawing_spec = connection_drawing_spec[connection] if isinstance(connection_drawing_spec, Mapping) else connection_drawing_spec
                cv2.line(image, Left_idx_to_coordinates[start_idx], Left_idx_to_coordinates[end_idx], drawing_spec.color, drawing_spec.thickness)

## Pose - Right Hand Connect
def pose_to_Rhand_draw_connections(image: np.ndarray,
                                 Rhand_landmark_list: landmark_pb2.NormalizedLandmarkList,
                                 pose_landmark_list: landmark_pb2.NormalizedLandmarkList,
                                 connections: Optional[List[Tuple[int, int]]] = None,
                                 connection_drawing_spec: Union[DrawingSpec, Mapping[Tuple[int, int], DrawingSpec]] = DrawingSpec()):

    if not Rhand_landmark_list and pose_landmark_list :
        return

    image_rows, image_cols, _ = image.shape
    Right_idx_to_coordinates = {}

    if Rhand_landmark_list:
        for idx, landmark in enumerate(Rhand_landmark_list.landmark):
            if ((landmark.HasField('visibility') and landmark.visibility < _VISIBILITY_THRESHOLD) or (landmark.HasField('presence') and landmark.presence < _PRESENCE_THRESHOLD)):
                continue
            landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y, image_cols, image_rows)
            if landmark_px and idx == 0:
                Right_idx_to_coordinates[0] = landmark_px

    for idx, landmark in enumerate(pose_landmark_list.landmark):
        if ((landmark.HasField('visibility') and landmark.visibility < _VISIBILITY_THRESHOLD) or (landmark.HasField('presence') and landmark.presence < _PRESENCE_THRESHOLD)):
            continue
        landmark_px = _normalized_to_pixel_coordinates(landmark.x, landmark.y, image_cols, image_rows)
        if landmark_px and idx == 14:
            Right_idx_to_coordinates[1] = landmark_px

    if connections:
        num_landmarks = len(Rhand_landmark_list.landmark) + len(pose_landmark_list.landmark)

        for connection in connections:
            start_idx = connection[0]
            end_idx = connection[1]
            if not (0 <= start_idx < num_landmarks and 0 <= end_idx < num_landmarks):
                raise ValueError(
                    f'Landmark index is out of range. Invalid connection 'f'from landmark #{start_idx} to landmark #{end_idx}.')
            if start_idx in Right_idx_to_coordinates and end_idx in Right_idx_to_coordinates:
                drawing_spec = connection_drawing_spec[connection] if isinstance(connection_drawing_spec, Mapping) else connection_drawing_spec
                cv2.line(image, Right_idx_to_coordinates[start_idx], Right_idx_to_coordinates[end_idx], drawing_spec.color, drawing_spec.thickness)

def Arm_draw_landmarks_connections(
        image: np.ndarray,
        leftHand_landmark_list: landmark_pb2.NormalizedLandmarkList,
        rightHand_landmark_list: landmark_pb2.NormalizedLandmarkList,
        pose_landmark_list: landmark_pb2.NormalizedLandmarkList,
        Hand_connections: Optional[List[Tuple[int, int]]] = None,
        pose_connections: Optional[List[Tuple[int, int]]] = None,
        posetohands_connections: Optional[List[Tuple[int, int]]] = None,
        Hand_landmark_drawing_spec: Union[DrawingSpec, Mapping[int, DrawingSpec]] = DrawingSpec(color=RED_COLOR),
        pose_landmark_drawing_spec: Union[DrawingSpec, Mapping[int, DrawingSpec]] = DrawingSpec(color=RED_COLOR),
        connection_drawing_spec: Union[DrawingSpec, Mapping[Tuple[int, int], DrawingSpec]] = DrawingSpec()):

    """
    hand_draw_landmarks_connections(image, leftHand_landmark_list, Hand_connections, Hand_landmark_drawing_spec, connection_drawing_spec)
    hand_draw_landmarks_connections(image, rightHand_landmark_list, Hand_connections, Hand_landmark_drawing_spec, connection_drawing_spec)
    pose_draw_landmarks_connections(image, pose_landmark_list, pose_connections, pose_landmark_drawing_spec, connection_drawing_spec)
    pose_to_Lhand_draw_connections(image, leftHand_landmark_list, pose_landmark_list, posetohands_connections, connection_drawing_spec)
    pose_to_Rhand_draw_connections(image, rightHand_landmark_list, pose_landmark_list, posetohands_connections, connection_drawing_spec)
    """
    hand_draw_landmarks_connections(image, leftHand_landmark_list, AC.NEW_HAND_CONNECTIONS, ds.get_default_hand_landmarks_style(), connection_drawing_spec)
    hand_draw_landmarks_connections(image, rightHand_landmark_list, AC.NEW_HAND_CONNECTIONS, ds.get_default_hand_landmarks_style(), connection_drawing_spec)
    pose_draw_landmarks_connections(image, pose_landmark_list, AC.NEW_POSE_CONNECTIONS, ds.get_default_pose_landmarks_style(), connection_drawing_spec)
    pose_to_Lhand_draw_connections(image, leftHand_landmark_list, pose_landmark_list, AC.POSETOHAND_CONNECTIONS, connection_drawing_spec)
    pose_to_Rhand_draw_connections(image, rightHand_landmark_list, pose_landmark_list, AC.POSETOHAND_CONNECTIONS, connection_drawing_spec)
