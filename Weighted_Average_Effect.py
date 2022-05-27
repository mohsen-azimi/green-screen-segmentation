import cv2
import numpy as np
from collections import deque

class AverageBuffer:
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)
        self.shape = None
    def apply(self, frame):
        self.shape = frame.shape
        self.buffer.append(frame)
    def get_frame(self):
        mean_frame = np.zeros(self.shape, dtype='float32')
        for item in self.buffer:
            mean_frame += item
        mean_frame /= len(self.buffer)
        return mean_frame.astype('uint8')

class WeightedAverageBuffer(AverageBuffer):
    def get_frame(self):
        mean_frame = np.zeros(self.shape, dtype='float32')
        i = 0
        for item in self.buffer:
            i += 4
            mean_frame += item*i
        mean_frame /= (i*(i + 1))/8.0
        return mean_frame.astype('uint8')

if __name__ == "__main__":
    # Setup camera
    cap = cv2.VideoCapture(1)
    # Set a smaller resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    average_buffer = AverageBuffer(30)
    weighted_buffer = WeightedAverageBuffer(30)
    while True:
        # Capture frame-by-frame
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, (640, 480))
        frame_f32 = frame.astype('float32')
        average_buffer.apply(frame_f32)
        weighted_buffer.apply(frame_f32)
        cv2.imshow('WebCam', frame)
        cv2.imshow("Average", average_buffer.get_frame())
        cv2.imshow("Weighted average", weighted_buffer.get_frame())
        if cv2.waitKey(1) == ord('q'):
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


