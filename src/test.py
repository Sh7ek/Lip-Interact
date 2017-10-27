import cv2
from src.ImageFPS import FPS
from src.ImageCamCaptureThread import WebCamVideoStream
import dlib
import imutils
from imutils import face_utils
import numpy as np

video_stream = WebCamVideoStream(src=0).start()
fps = FPS().start()

last_image = np.zeros(shape=(480, 640, 3), dtype=np.uint8);

# while True:
while True:
    image = video_stream.read()
    if np.array_equal(image, last_image):
        print("hello")
    if video_stream.notEmpty() and (not np.array_equal(image, last_image)):  # cannot be omitted, because the first frame may be empty on the camera
        last_image = image.copy()
        cv2.rectangle(image, (10, 10), (100, 100), (0, 255, 0), 2)
        cv2.imshow('frame', image)
        fps.update()
        # Press Q on keyboard to stop
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

video_stream.stop()
cv2.destroyAllWindows()
