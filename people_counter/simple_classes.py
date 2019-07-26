from enum import Enum


class ObjectDirection(Enum):
    UNKNOWN = 0
    ENTERING = 1
    EXITING = 2


class CameraType(Enum):
    UNKNOWN = 0
    ANDROID_CAMON = 1
    MAC_WEBCAM = 2
    SURVEILLANCE = 3


class InputSource:
    def __init__(self, camera_type: CameraType = CameraType.UNKNOWN,
                 location: str = 'Unknown',
                 monitored_direction: ObjectDirection = ObjectDirection.UNKNOWN):
        self.cameraType = camera_type
        self.location = location
        self.monitoredDirection = monitored_direction
