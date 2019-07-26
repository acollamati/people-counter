import cv2
import numpy as np
from people_counter.simple_classes import InputSource
from people_counter.simple_classes import ObjectDirection, CameraType
from people_counter.input_frame import InputFrame
from people_counter.frame_processor import FrameProcessor
from people_counter.face_detection import FaceDetectionEvent
from mjpeg.client import MJPEGClient
import time
from os import listdir, mkdir
import face_recognition
import shutil
from typing import List
import logging


class Session:
    """
    Acquire streaming and analyze images
    """

    def __init__(self, video_input='webcam',
                 location="Unknown Location",
                 max_visit_time_hours=4,
                 encoding_path="./encodings/",
                 images_path="./images/"):
        """

        :param video_input: URL, "webcam" or video file
        :param location: Where is the camera?
        :param max_visit_time_hours: On average, how long a visit last?
        :param encoding_path: dir with slashes, eg "./encodings/". Subdirs are names of past encoded faces
        :param images_path: dir with dirs of pictures of known people. Subdirs are names
        """

        self.location = location

        if video_input[:4] == 'http':
            logging.info("Session init: using Android at " + str(video_input))
            self.camera_type = CameraType.ANDROID_CAMON
            self.source = InputSource(self.camera_type, self.location, ObjectDirection.ENTERING)
            self.client = MJPEGClient(video_input)

        elif video_input == 'webcam':
            logging.info("Session init: using Webcam.")
            self.camera_type = CameraType.MAC_WEBCAM
            self.source = InputSource(self.camera_type, self.location, ObjectDirection.ENTERING)
            self.video_capture = cv2.VideoCapture(0)
        else:
            self.video_capture = cv2.VideoCapture(video_input)
            if self.video_capture.isOpened():
                logging.info("Opened file " + str(video_input))
                self.camera_type = CameraType.SURVEILLANCE
                self.source = InputSource(self.camera_type, self.location, ObjectDirection.ENTERING)
                pass
            else:
                raise ValueError("'video_input' must be either valid url, video file or 'webcam'")
        self.images_path = images_path
        self.__resize_factor = 1.0
        self.images = []
        self.known_face_names = []

        self.encoding_path = encoding_path
        self.imported_events: List[FaceDetectionEvent] = []
        self.max_visit_time_hours = max_visit_time_hours

        self.load_known_faces()  # put in encoding_path the faces in images_path
        # load a list of all past encoding, with timestamp = (now - visit_time)
        self.load_imported_events()
        self.processor = FrameProcessor(self.source,
                                        self.max_visit_time_hours,
                                        self.imported_events)

    def load_known_faces(self):
        """Load images in the dir images_path into images and the
        name into known_face_names.
        Images are in directories with the name of person (so we can
        have more images per person).
        """
        logging.info("Loading known faces.")
        if not shutil.os.access(self.images_path, 400):
            logging.info("load_known_faces: Image directory " + self.images_path + " not found, nothing added.")
            return None
        for d in listdir(self.images_path):
            if d[0] == '.':
                continue
            # Load known pictures and name
            path = self.images_path + d + "/"
            for image in listdir(path):
                if image[0] == '.':
                    continue
                self.known_face_names.append(d)
                self.images.append(face_recognition.load_image_file(path + "/" + image))

        for i, name in enumerate(self.known_face_names):
            try:
                mkdir(self.encoding_path + name)
            except FileExistsError:
                shutil.rmtree(self.encoding_path + name)
                mkdir(self.encoding_path + name)
            enc = face_recognition.face_encodings(self.images[i])[0]
            np.save(self.encoding_path + name + "/" + FaceDetectionEvent.make_id(enc), enc)
            logging.info("load_known_faces: saved image of " + str(name) + " in " + str(self.encoding_path))

    def load_imported_events(self):
        """Fill imported_events with all encodings in the
            encodings directory (which has all images from famous
            people and from past customers - if it's not from scratch).

            TODO Eventually it can re-written to get data from DB
        """
        logging.info("Loading imported event from " + str(self.encoding_path))
        for name in listdir(self.encoding_path):
            if name[0] == '.':
                continue
            for enc in listdir(self.encoding_path + name):
                if enc[0] == '.':
                    continue
                ev = FaceDetectionEvent(location=[0, 0, 0, 0],
                                        encoding=np.load((self.encoding_path + name + "/" + enc)),
                                        timestamp=int(time.time() * 1000 -
                                                      3600000*self.max_visit_time_hours),
                                        input_source=InputSource(),
                                        name=str(name))
                self.imported_events.append(ev)
                logging.info("Session.load_imported_events: appended past event. Encoding " + str(enc) +
                             "in directory " + name)
        if len(self.imported_events) == 0:
            logging.info("Session.load_imported_events: no past events found")

    def __start_mjpeg_client(self):
        # Allocate memory buffers for frames
        bufs = self.client.request_buffers(1000000, 50)
        for b in bufs:
            self.client.enqueue_buffer(b)
        # Start the client in a background thread
        self.client.start()

    def __get_frame(self):
        if self.camera_type == CameraType.MAC_WEBCAM:
            # Grab a single frame of video
            ret, frame = self.video_capture.read()
            # convert frame from BGR to RGB (openCV uses BGR)
            rgb_frame = frame[:, :, ::-1]

            # Resize frame of video to 1/4 size for faster face recognition processing
            return cv2.resize(rgb_frame, (0, 0),
                              fx=self.__resize_factor,
                              fy=self.__resize_factor)

        else:
            # fetch jpeg frame from client queue
            buf = self.client.dequeue_buffer()
            jpeg_data = buf.data[:buf.used]
            x = np.frombuffer(jpeg_data, dtype='uint8')
            # decode jpeg
            # TODO check if b&w is better, so we get rid of BGR-RGB stuff
            frame = cv2.imdecode(x, 1)
            # convert frame from BGR to RGB (openCV uses BGR)
            rgb_frame = frame[:, :, ::-1]

            self.client.enqueue_buffer(buf)

            return cv2.resize(rgb_frame, (0, 0),
                              fx=self.__resize_factor,
                              fy=self.__resize_factor)

    @staticmethod
    def draw_overlay(cv_frame, ev: FaceDetectionEvent):
        top = ev.boundingRect.y
        right = ev.boundingRect.x + ev.boundingRect.w
        bottom = ev.boundingRect.y + ev.boundingRect.h
        left = ev.boundingRect.x

        top *= 1  # TODO put self.resize_factor
        right *= 1
        bottom *= 1
        left *= 1

        # Draw a box around the face
        cv2.rectangle(cv_frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(cv_frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(cv_frame, ev.name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    def run(self):

        if self.camera_type == CameraType.ANDROID_CAMON:
            self.__start_mjpeg_client()

        f = open("face_detected.log", "a+")

        # TODO:
        #  webcam: while(video_capture.isOpened())
        #  ipcam: ?
        skip_images = 2  # every other skip_images images are analysed
        image_number = 0
        while True:
            image_number += 1
            if image_number % skip_images == 0:
                continue

            current_raw_frame = self.__get_frame()
            input_frame = InputFrame(current_raw_frame,
                                     self.source,
                                     time.time() * 1000)

            events = self.processor.process(input_frame)

            for index, ev in enumerate(events):

                print(ev.to_csv())
                f.write(ev.to_csv() + "\n")
                f.flush()
                if ev.virgin:
                    mkdir(self.encoding_path + str(ev.name))
                    np.save(self.encoding_path + str(ev.name) + "/" + ev.id, ev.encoding)

            # Display the results for webcam
            for ev in events:
                Session.draw_overlay(current_raw_frame, ev)

        f.close()
    # Display the resulting image
    #cv2.imshow('Video', currentRawFrame)

    # Hit 'q' on the keyboard to quit!
    #if cv2.waitKey(1) & 0xFF == ord('q'):
    #    break


#video_capture.release()
        #cv2.destroyAllWindows()


logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
url = 'http://192.168.1.74:8080/video/mjpeg'  #applicativo android camon
url = 'http://192.168.1.2:8080/video/mjpeg'  #applicativo android camon
#url = 'http://192.168.4.2:8000/camera/mjpeg'  #applicativo pc cam2web from code project


session = Session(video_input="webcam")

session.run()
