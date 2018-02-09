import cv2
from src.ImageFPS import FPS
from src.ImageCamCaptureThread import WebCamVideoStream
import dlib
import imutils
from imutils import face_utils
import numpy as np
import math
from collections import deque
from src.ImageLipNetRecognizeThread import RecognizeThread
from src.SocketServer import SocketServer

video_stream = WebCamVideoStream(src=0).start()
FPS_STARTED = False


cv2.namedWindow("frame");
cv2.moveWindow("frame", 550, 450);

last_image = np.zeros(shape=(480, 640, 3), dtype=np.uint8);
if __name__ == '__main__':
    # while True:
    fps = FPS().start()
    while True:
        image = video_stream.read()
        if video_stream.notEmpty() and (not np.array_equal(image, last_image)):  # cannot be omitted, because the first frame may be empty on the camera
            last_image = image.copy()
            cv2.imshow("frame", image)
            fps.update()
            # Press Q on keyboard to stop
            # Press F on keyboard to force face re-detection on the next frame
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                force_detect_face = True

    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    video_stream.stop()
    cv2.destroyAllWindows()

