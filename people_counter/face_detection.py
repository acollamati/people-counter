import datetime
import time
from people_counter.simple_classes import InputSource
import uuid
import logging
import hashlib


class FaceBoundingRect:
    def __init__(self, location):
        # relative face position inside image tuple in (top, right, bottom, left)
        top = location[0]
        right = location[1]
        bottom = location[2]
        left = location[3]
        self.x = left
        self.y = top
        self.w = right - left
        self.h = bottom - top

    def __str__(self):
        return "[" + str(self.x) + ":" + str(self.y) + "](" + str(self.w) + ":" + str(self.h) + ")"

    def to_csv(self):
        return str(self.x) + "," + str(self.y) + "," + str(self.w) + "," + str(self.h)


class FaceDetectionEvent:

    def __init__(self, location,
                 encoding,
                 timestamp,
                 input_source: InputSource = None,
                 name=None,
                 virgin: bool = None):
        # relative face position inside image tuple in (top, right, bottom, left)
        self.boundingRect = FaceBoundingRect(location)
        # binary data embedding the face features  128 double array Numpy
        self.encoding = encoding
        # time from epoch in ms
        self.timestamp = timestamp
        # camera string id
        self.inputSource = input_source
        self.id = self.make_id(encoding)
        if name is None:
            self.name = str(uuid.uuid1())
        else:
            self.name = name
        self.virgin = virgin

    def __str__(self):
        dt = datetime.datetime.fromtimestamp(self.timestamp / 1000).strftime('%c')
        s = dt + ":Object " + self.inputSource.location + ":Pos  " + str(self.boundingRect)
        return s

    def update(self, encoding=None, name=None, virgin=None):
        if encoding is not None:
            self.encoding = encoding
            self.id = self.make_id(encoding)
        if name is not None:
            self.name = name
        if virgin is not None:
            self.virgin = virgin

    def to_csv(self):
        return str(int(self.timestamp)) + "," + self.id + "," + self.boundingRect.to_csv() + "," + self.inputSource.location

    @staticmethod
    def create_event(location, encoding, camera_id):
        return FaceDetectionEvent(location, encoding, time.time() * 1000, camera_id)

    @staticmethod
    def make_id(e):
        return hashlib.md5(str.encode(str(e))).hexdigest()


class FaceDetectionStats:
    def __init__(self, first_det_time: int, encoding,
                 n_occurences_since_first_det_time: int):
        self.encoding = encoding
        self.firstDetTime = first_det_time
        self.nOccurencesSinceFirstDetTime = n_occurences_since_first_det_time
