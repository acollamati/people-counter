import face_recognition
import numpy as np
from people_counter.simple_classes import InputSource
from people_counter.input_frame import InputFrame
from people_counter.face_detection import FaceDetectionEvent
from typing import List
import logging


class FrameProcessor:

    def __init__(self, source: InputSource,
                 max_visit_time_hours,
                 imported_events: List[FaceDetectionEvent],
                 tolerance=0.6):
        """

        :param source:
        :param max_visit_time_hours: how long a visit to the store might last
        :param imported_events: previously seen events (eg found in encoding_path)
        :param tolerance: for comparing faces
        """
        self.source = source
        # encoding_buffer makes sure we only return a face if it has seen in a few
        # consecutive frames.
        self.encoding_buffer = np.array([], np.ndarray)
        self.imported_events = imported_events
        self.recorded_events: List[FaceDetectionEvent] = []
        # consider a new visit only if it happens
        # after max_visit_time_hours from the first time
        self.max_visit_time_hours = max_visit_time_hours
        self.tolerance = tolerance  # faces are considered the same if distance < tolerance

    def process(self, frame: InputFrame) -> List[FaceDetectionEvent]:
        events: List[FaceDetectionEvent] = []
        # Find all the faces and face encodings in the current frame of video

        # apply faceDetection algorithm
        # TODO check if face_locations slows down (if we don't have to show the square,
        # TODO maybe is better not to do face_locations)
        face_locations = face_recognition.face_locations(frame.frame)
        print("I found {} face(s) in this photograph.".format(len(face_locations)))
        face_encodings = face_recognition.face_encodings(face_image=frame.frame,
                                                         known_face_locations=face_locations,
                                                         num_jitters=2  # times to re-sample
                                                         )

        # append detected faces to lastEvent list
        for index, face_encoding in enumerate(face_encodings):
            ev = FaceDetectionEvent.create_event(face_locations[index],
                                                 face_encoding, self.source)

            filtered_event = self.process_single_event(ev)
            if filtered_event is not None:
                events.append(filtered_event)
                
        return events

    def process_single_event(self, ev: FaceDetectionEvent) -> FaceDetectionEvent:
        """

        :param ev:
        :return:
        """
        print(">>>>>>>>  encoding_buffer shape: " + str(self.encoding_buffer.shape))
        result = None
        found_recorded = False
        found_imported = False
        # Check if a similar face was recorded in this session
        if len(self.recorded_events) > 0:
            logging.debug("FrameProcessor.process_single_event: recorded_events -> " + str(len(self.recorded_events)))
            recorded_encodings = [fds.encoding for fds in self.recorded_events]
            recorded_distances = face_recognition.face_distance(recorded_encodings, ev.encoding)
            best_match_recorded_idx = np.argmin(recorded_distances)
            if recorded_distances[best_match_recorded_idx] < self.tolerance:
                logging.debug("FrameProcessor.process_single_event: found events close to a recently recorded one")
                found_recorded = True

        # Check if a similar face was recorded in the past
        if len(self.imported_events) > 0:
            logging.debug("imported_events: " + str(len(self.imported_events)))
            imported_encodings = [fds.encoding for fds in self.imported_events]
            imported_distances = face_recognition.face_distance(imported_encodings, ev.encoding)
            best_match_imported_idx = np.argmin(imported_distances)  # TODO??? PyCharm says it's not an int
            if imported_distances[best_match_imported_idx] < self.tolerance:
                logging.debug("FrameProcessor.process_single_event: found events close to past")
                found_imported = True

        # recorded is more similar than imported
        if ((found_imported and found_recorded and
            recorded_distances[best_match_recorded_idx] <= imported_distances[best_match_imported_idx]) or
                (found_recorded and not found_imported)):
            logging.debug("process_single_event: return __process_recorded_event bc recorded won or no imported")
            # customer was already caught on camera since we started recording
            ev.update(encoding=recorded_encodings[best_match_recorded_idx],
                      name=self.recorded_events[best_match_recorded_idx].name)
            result = self.__process_recorded_event(ev, best_match_recorded_idx)
        # imported more similar than recorded
        elif ((found_imported and found_recorded and
                recorded_distances[best_match_recorded_idx] > imported_distances[best_match_imported_idx])
                or (not found_recorded and found_imported)):
            ev.update(encoding=imported_encodings[best_match_imported_idx],
                      name=self.imported_events[best_match_imported_idx].name)
            self.recorded_events.append(ev)
            logging.debug("FrameProcessor.process_single_event: return imported event bc imported won or no recorded")
            result = ev
        else:
            logging.debug("process_single_event: Event is new.")
            # If in buffer: put in recorded, delete from buffer, return it
            # If not: put in buffer
            logging.debug("FrameProcessor.process_single_event: encoding_buffer shape is " + str(self.encoding_buffer.shape))
            if len(self.encoding_buffer) > 0:
                buffer_similar = np.asarray(face_recognition.compare_faces(known_face_encodings=self.encoding_buffer,
                                                                face_encoding_to_check=ev.encoding,
                                                                tolerance=self.tolerance))
                if buffer_similar.any():  # any True ?
                    logging.debug("FrameProcessor.process_single_event: Found similar face in buffer")
                    self.recorded_events.append(ev)
                    # 'ev' went to recorded, so the old faces of 'ev' can be deleted
                    # in the future "recorded_event" will decide if this face is good or not.
                    self.encoding_buffer = np.delete(self.encoding_buffer, np.where(buffer_similar), 0)
                    logging.debug("FrameProcessor.process_single_event: encoding_buffer after deletion is " + str(
                        self.encoding_buffer.shape))

                    ev.update(virgin=True)
                    result = ev
                else:
                    logging.debug("FrameProcessor.process_single_event: Found NO similar faces in buffer, adding")
                    self.encoding_buffer = np.append(self.encoding_buffer, [ev.encoding], 0)
            else:
                logging.debug("Empty buffer, adding")
                self.encoding_buffer = np.asarray([ev.encoding])
        logging.debug("RETURNG " + str(result))
        return result

    def __process_recorded_event(self, ev, idx):
        """If the event happens after max_visit_time_hours from the
        first detection, return the event. Otherwise do nothing.

        :param ev
        :param idx index of the event in recorded_events with matching encoding
        """
        # check date time
        delta_ms = ev.timestamp - self.recorded_events[idx].timestamp
        # pass enough time since first detection. I can retrigger event
        if delta_ms > self.max_visit_time_hours*86400*1000:
            return ev
        else:
            # event is dropped because is to close to the previous one
            return None
