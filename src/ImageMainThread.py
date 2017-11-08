import cv2
from src.ImageFPS import FPS
from src.ImageCamCaptureThread import WebCamVideoStream
import dlib
import imutils
from imutils import face_utils
import numpy as np
import math
from collections import deque

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../resource/shape_predictor_68_face_landmarks.dat")

video_stream = WebCamVideoStream(src=0).start()
fps = FPS().start()

last_image = np.zeros(shape=(480, 640, 3), dtype=np.uint8);
force_detect_face = True
face_rect = dlib.rectangle(left=0, top=0, right=0, bottom=0)
shape_margin = 10
face_limit = 200

gap_feature_list = []
gap_window_len = 7
gap_max_limit = 0.1
gap_max_update_alpha = 0.5
gap_std_limit = 0.02
isSpeaking = False
gapDeque = deque([], maxlen=gap_window_len)

stable_mouth_crop_left = 0
stable_mouth_crop_right = 0
stable_mouth_crop_width = 0
stable_mouth_crop_top = 0
stable_mouth_crop_height = 0
HORIZANTAL_PAD_RATIO = 0.1
MOUTH_CROP_WIDTH = 100
MOUTH_CROP_HEIGHT = 80
normalize_ratio = None
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
            (shape_x, shape_y, shape_w, shape_h) = cv2.boundingRect(shape_array) # new face region
            (mouth_x, mouth_y, mouth_w, mouth_h) = cv2.boundingRect(lip_array)  # mouth region
            (face_x, face_y, face_w, face_h) = face_utils.rect_to_bb(face_rect)  # last face region

            # first use gap_feature to check is the user is speaking
            # gap_distance: mean distance between lip inner points
            gap_distance = (math.hypot(lip_array[13][0]-lip_array[19][0], lip_array[13][1]-lip_array[19][1]) +
                            math.hypot(lip_array[14][0] - lip_array[18][0], lip_array[14][1] - lip_array[18][1]) +
                            math.hypot(lip_array[15][0] - lip_array[17][0], lip_array[15][1] - lip_array[17][1]))/3
            # width_distance: mean distance between lip inner corner points
            width_distance = math.hypot(lip_array[12][0] - lip_array[16][0], lip_array[12][1] - lip_array[16][1])
            # use their ratio to detect if is speaking or not
            gap_feature = gap_distance/width_distance
            gap_feature_list.append(gap_feature)
            # print("gap feature: {}".format(gap_feature))
            gapDeque.append(gap_feature)
            if len(gapDeque) == gapDeque.maxlen:
                maxGap = max(gapDeque)
                meanGap = sum(gapDeque) / gapDeque.maxlen
                stdGap = math.sqrt(sum([(x - meanGap) ** 2 for x in gapDeque]) / gapDeque.maxlen)
                # If the user is not speaking
                if not isSpeaking:
                    # If start to speak
                    if gap_feature > max(gap_max_limit * 1.5, 0.1):  # maybe is starting speaking, stop updating max_limit
                        isSpeaking = True
                        # start_speaking_index = i
                    # If still not speaking, update some parameters
                    elif maxGap < max(gap_max_limit, 0.1) and stdGap < gap_std_limit:
                        gap_max_limit = gap_max_limit * (1 - gap_max_update_alpha) + meanGap * 2 * gap_max_update_alpha
                        # get mouth crop image on original image
                        stable_mouth_crop_left = int(mouth_x - mouth_w * HORIZANTAL_PAD_RATIO)
                        stable_mouth_crop_right = int(mouth_x + mouth_w * (1 + HORIZANTAL_PAD_RATIO))
                        stable_mouth_crop_width = stable_mouth_crop_right - stable_mouth_crop_left
                        stable_mouth_crop_height = int(round(stable_mouth_crop_width * 0.8))
                        stable_mouth_crop_top = int(round(mouth_y + mouth_h/2 - stable_mouth_crop_height / 3))
                # If the user is speaking
                else:
                    # If the user stops speaking
                    if maxGap < min(gap_max_limit, 0.1) and stdGap < gap_std_limit * 0.5:  # has stopped speaking
                        # end_speaking_index = i
                        isSpeaking = False
                        gap_max_limit = meanGap * 2
                        stable_mouth_crop_left = int(mouth_x - mouth_w * HORIZANTAL_PAD_RATIO)
                        stable_mouth_crop_right = int(mouth_x + mouth_w * (1 + HORIZANTAL_PAD_RATIO))
                        stable_mouth_crop_width = stable_mouth_crop_right - stable_mouth_crop_left
                        stable_mouth_crop_height = int(round(stable_mouth_crop_width * 0.8))
                        stable_mouth_crop_top = int(round(mouth_y + mouth_h / 2 - stable_mouth_crop_height / 3))

            # estimate the next face_rect by shape of this frame to save time
            face_rect = dlib.rectangle(
                left=shape_x - shape_margin,
                top=shape_y - shape_margin,
                right=shape_x + shape_w + shape_margin,
                bottom=shape_y + shape_h - shape_margin)
            if face_rect.right() - face_rect.left() < face_limit or face_rect.bottom() - face_rect.top() < face_limit:
                force_detect_face = True

            if stable_mouth_crop_width > 0:
                # get the normalized mouth crop image
                normalize_ratio = MOUTH_CROP_WIDTH / float(stable_mouth_crop_width)  # generally < 1
                normalized_image = cv2.resize(image, None, fx=normalize_ratio, fy=normalize_ratio, interpolation=cv2.INTER_AREA)
                normalized_mouth_crop_left = int(round(stable_mouth_crop_left * normalize_ratio))
                normalized_mouth_crop_top = int(round(stable_mouth_crop_top * normalize_ratio))
                normalized_mouth_crop_image = normalized_image[normalized_mouth_crop_top:normalized_mouth_crop_top + MOUTH_CROP_HEIGHT,
                                              normalized_mouth_crop_left:normalized_mouth_crop_left + MOUTH_CROP_WIDTH]
                # print(normalized_mouth_crop_image.shape)
                cv2.imshow('mouth_crop', normalized_mouth_crop_image)

            # # Display the resulting frame
            cv2.rectangle(image, (face_x, face_y), (face_x + face_w, face_y + face_h), (0, 255, 0), 2)
            # print("{} {} {} {}".format(stable_mouth_crop_left, stable_mouth_crop_top, stable_mouth_crop_width, stable_mouth_crop_height))
            if isSpeaking:
                # cv2.rectangle(image, (mouth_x, mouth_y), (mouth_x + mouth_w, mouth_y + mouth_h), (255, 0, 0), 2)
                cv2.rectangle(image, (stable_mouth_crop_left, stable_mouth_crop_top),
                              (stable_mouth_crop_left + stable_mouth_crop_width,
                               stable_mouth_crop_top + stable_mouth_crop_height), (0, 0, 255), 2)
            else:
                # cv2.rectangle(image, (mouth_x, mouth_y), (mouth_x + mouth_w, mouth_y + mouth_h), (0, 255, 0), 2)
                cv2.rectangle(image, (stable_mouth_crop_left, stable_mouth_crop_top),
                              (stable_mouth_crop_left + stable_mouth_crop_width,
                               stable_mouth_crop_top + stable_mouth_crop_height), (0, 255, 0), 2)
            for (x, y) in lip_array[12:20]:
                cv2.circle(image, (x, y), 2, (255, 255, 255), -1)
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

file = open("../resource/gap_feature.txt", "w")
for g in gap_feature_list:
    file.write(str(g) + "\n")
file.close()
