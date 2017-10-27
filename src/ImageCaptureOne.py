import cv2
from src.ImageFPS import FPS
from src.ImageCamCaptureThread import WebCamVideoStream
import numpy as np


video_stream = WebCamVideoStream(src=0).start()
fps = FPS().start()

last_image = np.zeros(shape=(480, 640, 3), dtype=np.uint8);
force_detect_face = True

# while True:
while True:
    image = video_stream.read()
    if video_stream.notEmpty():  # cannot be omitted, because the first frame may be empty on the camera

        # Display the resulting frame
        cv2.imshow('frame', image)
        fps.update()
        # Press Q on keyboard to stop recording
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.imwrite("../resource/people.jpg", image)
            break

fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

video_stream.stop()
cv2.destroyAllWindows()
