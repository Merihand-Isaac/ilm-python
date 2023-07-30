import mediapipe.python.solutions.hands_connections as hc
import mediapipe.python.solutions.pose_connections as pc

# Add Hand Connections
AddHandConnections = ((0, 9), (0, 13), (0, 17), (1, 5))
NEW_HAND_CONNECTIONS = frozenset().union(*[hc.HAND_PALM_CONNECTIONS, hc.HAND_THUMB_CONNECTIONS,
                                           hc.HAND_INDEX_FINGER_CONNECTIONS, hc.HAND_MIDDLE_FINGER_CONNECTIONS,
                                           hc.HAND_RING_FINGER_CONNECTIONS, hc.HAND_PINKY_FINGER_CONNECTIONS, AddHandConnections])

# Delete Pose Connections
tempPoseConnections = list(pc.POSE_CONNECTIONS)
for idx, array in enumerate(tempPoseConnections) :
    if array == (11, 12) :
        del tempPoseConnections[idx]
NEW_POSE_CONNECTIONS = frozenset(tuple(tempPoseConnections))

# Add Pose to Hand Connections
POSETOHAND_CONNECTIONS = frozenset([(0, 1)])