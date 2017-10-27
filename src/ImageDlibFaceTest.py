from imutils import  face_utils
import numpy as np
import imutils
import dlib
import cv2
import datetime

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../resource/shape_predictor_68_face_landmarks.dat")

image = cv2.imread("../resource/people.jpg")

start = datetime.datetime.now()
rects = detector(image)
end = datetime.datetime.now()
print("face dectect time: {}".format((end - start).total_seconds()))

if len(rects) > 0:
    start = datetime.datetime.now()
    shape_array = face_utils.shape_to_np(predictor(image, rects[0]))
    end = datetime.datetime.now()
    print("landmark dectect time: {}".format((end - start).total_seconds()))
    (shape_x, shape_y, shape_w, shape_h) = face_utils.rect_to_bb(rects[0])
    cv2.rectangle(image, (shape_x, shape_y), (shape_x+shape_w, shape_y+shape_h), (0, 255, 0), 2)

    for (x, y) in shape_array:
        cv2.circle(image, (x, y), 2, (255, 255, 0))

cv2.imshow("output", image)
cv2.waitKey(0)