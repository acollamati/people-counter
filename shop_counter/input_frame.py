from shop_counter.input_source import InputSource


class InputFrame:

    def __init__(self, frame, input_source: InputSource, time_stamp):
        self.frame = frame
        self.inputSource = input_source
        self.timeStamp = time_stamp

