import cv2
from src.ImageFPS import FPS
from src.ImageCamCaptureThread import WebCamVideoStream
import dlib
import imutils
from imutils import face_utils
import numpy as np

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../resource/shape_predictor_68_face_landmarks.dat")

video_stream = WebCamVideoStream(src=0).start()
fps = FPS().start()

last_image = np.zeros(shape=(480, 640, 3), dtype=np.uint8);
force_detect_face = True
face_rect = dlib.rectangle(left=0, top=0, right=0, bottom=0)
shape_margin = 26
face_limit = 200

# while True:
while True:
    image = video_stream.read()
    if video_stream.notEmpty() and (not np.array_equal(image, last_image)):  # cannot be omitted, because the first frame may be empty on the camera
        last_image = image.copy()
        # when we need to detect the face with dlib
        if force_detect_face:
            rects = detector(image)
            if len(rects) == 1:
                face_rect = rects[0]
                force_detect_face = False
        # we have got an face rect
        if not force_detect_face:
            # detect the face and the landmarks
            shape_array = face_utils.shape_to_np(predictor(image, face_rect))  # ndarray (68, 2)
            lip_array = shape_array[48:68]  # mouth index [48: 68)
            # get the bounds of landmarks
            (shape_x, shape_y, shape_w, shape_h) = cv2.boundingRect(shape_array)
            (mouth_x, mouth_y, mouth_w, mouth_h) = cv2.boundingRect(lip_array)
            (face_x, face_y, face_w, face_h) = face_utils.rect_to_bb(face_rect)
            # estimate the next face_rect by shape of this frame to save time
            face_rect = dlib.rectangle(
                left=shape_x - shape_margin,
                top=shape_y - shape_margin,
                right=shape_x + shape_w + shape_margin,
                bottom=shape_y + shape_h - shape_margin)
            if face_rect.right() - face_rect.left() < face_limit or face_rect.bottom() - face_rect.top() < face_limit:
                force_detect_face = True
            # # Display the resulting frame
            cv2.rectangle(image, (face_x, face_y), (face_x + face_w, face_y + face_h), (0, 255, 0), 2)
            cv2.rectangle(image, (mouth_x, mouth_y), (mouth_x + mouth_w, mouth_y + mouth_h), (0, 255, 0), 2)
            fps.update()

        cv2.imshow('frame', image)
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
