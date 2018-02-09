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
from threading import Thread
import atexit

RECOGNIZER = True
USER_ADJUSTED = False  # always false at first

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../resource/shape_predictor_68_face_landmarks.dat")

video_stream = WebCamVideoStream(src=0).start()
FPS_STARTED = False

socketServer = SocketServer(serverIP='192.168.1.100', serverPort=6000)
socketServer.Start()

if RECOGNIZER:
    recognizeThread = RecognizeThread(socketServer=socketServer).start()

cv2.namedWindow("frame");
cv2.moveWindow("frame", 550, 450);

last_image = np.zeros(shape=(480, 640, 3), dtype=np.uint8);
force_detect_face = True
face_rect = dlib.rectangle(left=0, top=0, right=0, bottom=0)
shape_margin_h = 6
shape_margin_v = 12
face_limit = 200

# gap_feature_list = []
gap_window_len = 7  # 7 last day
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
stable_mouth_crop_vcenter = 0
stable_window_len = 6
stable_mouth_crop_left_deque = deque([], maxlen=stable_window_len)
stable_mouth_crop_right_deque = deque([], maxlen=stable_window_len)
stable_mouth_crop_top_deque = deque([], maxlen=stable_window_len)
stable_mouth_crop_vcenter_deque = deque([], maxlen=stable_window_len)

HORIZANTAL_PAD_RATIO = 0.15
MOUTH_CROP_WIDTH = 100
MOUTH_CROP_HEIGHT = 80
normalize_ratio = None

speakingNormalizedCropMouthList = []
speakingState = 0  # 0: not speaking stable, 1: detect start to speak 2: speaking 3: detect stop speaking
noSpeakingNormalizedCropMouthDeque = deque([], maxlen=gap_window_len)

speaking_length = 0

def goodbye():
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    video_stream.stop()
    if RECOGNIZER:
        recognizeThread.stop()
    socketServer.stop()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    atexit.register(goodbye)
    # while True:
    while True:
        image = video_stream.read()
        if video_stream.notEmpty() and (not np.array_equal(image, last_image)):  # cannot be omitted, because the first frame may be empty on the camera
            if RECOGNIZER and recognizeThread.model_loaded and not FPS_STARTED:
                fps = FPS().start()
                FPS_STARTED = True

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
                # gap_feature_list.append(gap_feature)
                # print("gap feature: {}".format(gap_feature))
                gapDeque.append(gap_feature)
                if len(gapDeque) == gapDeque.maxlen:
                    maxGap = max(gapDeque)
                    meanGap = sum(gapDeque) / gapDeque.maxlen
                    stdGap = math.sqrt(sum([(x - meanGap) ** 2 for x in gapDeque]) / gapDeque.maxlen)
                    # If the user is not speaking
                    if not isSpeaking:
                        # If start to speak
                        if gap_feature > max(gap_max_limit * 1.75, 0.12):  # maybe is starting speaking, stop updating max_limit
                            isSpeaking = True
                            speakingState = 1
                            Thread(target=socketServer.SendToAllConnections, args=('MOUTHOPEN\n',)).start()

                            # start_speaking_index = i
                        # If still not speaking, update some parameters
                        elif maxGap < max(gap_max_limit, 0.12) and stdGap < gap_std_limit:
                            gap_max_limit = gap_max_limit * (1 - gap_max_update_alpha) + meanGap * 2 * gap_max_update_alpha
                            speakingState = 0
                            # get mouth crop image on original image
                            stable_mouth_crop_left_temp = int(mouth_x - mouth_w * HORIZANTAL_PAD_RATIO)
                            stable_mouth_crop_right_temp = int(mouth_x + mouth_w * (1 + HORIZANTAL_PAD_RATIO))
                            stable_mouth_crop_vcenter_temp = mouth_y + mouth_h/2
                            stable_mouth_crop_left_deque.append(stable_mouth_crop_left_temp)
                            stable_mouth_crop_right_deque.append(stable_mouth_crop_right_temp)
                            stable_mouth_crop_vcenter_deque.append(stable_mouth_crop_vcenter_temp)

                            # stable_mouth_crop_left = int(sum(stable_mouth_crop_left_deque)/len(stable_mouth_crop_left_deque))
                            # stable_mouth_crop_right = int(sum(stable_mouth_crop_right_deque) / len(stable_mouth_crop_right_deque))
                            stable_mouth_crop_left = stable_mouth_crop_left_deque[0]
                            stable_mouth_crop_right = stable_mouth_crop_right_deque[0]
                            stable_mouth_crop_vcenter = stable_mouth_crop_vcenter_deque[0]
                            stable_mouth_crop_width = stable_mouth_crop_right - stable_mouth_crop_left
                            stable_mouth_crop_height = int(round(stable_mouth_crop_width * float(MOUTH_CROP_HEIGHT)/float(MOUTH_CROP_WIDTH)))
                            stable_mouth_crop_top = int(round(stable_mouth_crop_vcenter - stable_mouth_crop_height * 0.36))
                    # If the user is speaking
                    else:
                        # If the user stops speaking
                        if (maxGap < min(gap_max_limit, 0.12) or maxGap < 0.04) and stdGap < gap_std_limit*0.6:  # has stopped speaking
                            # end_speaking_index = i
                            isSpeaking = False
                            gap_max_limit = meanGap * 2
                            speakingState = 3
                            # get mouth crop image on original image
                            stable_mouth_crop_left = int(mouth_x - mouth_w * HORIZANTAL_PAD_RATIO)
                            stable_mouth_crop_right = int(mouth_x + mouth_w * (1 + HORIZANTAL_PAD_RATIO))
                            stable_mouth_crop_width = stable_mouth_crop_right - stable_mouth_crop_left
                            stable_mouth_crop_height = int(round(stable_mouth_crop_width * float(MOUTH_CROP_HEIGHT)/float(MOUTH_CROP_WIDTH)))
                            stable_mouth_crop_vcenter = mouth_y + mouth_h/2
                            stable_mouth_crop_top = int(round(stable_mouth_crop_vcenter- stable_mouth_crop_height * 0.36))

                            stable_mouth_crop_left_deque.clear()
                            stable_mouth_crop_right_deque.clear()
                            stable_mouth_crop_vcenter_deque.clear()
                            stable_mouth_crop_left_deque.append(stable_mouth_crop_left)
                            stable_mouth_crop_right_deque.append(stable_mouth_crop_right)
                            stable_mouth_crop_vcenter_deque.append(stable_mouth_crop_vcenter)
                            Thread(target=socketServer.SendToAllConnections, args=('MOUTHCLOSE\n',)).start()
                        # the user continues speaking
                        else:
                            speakingState = 2

                # estimate the next face_rect by shape of this frame to save time
                face_rect = dlib.rectangle(
                    left=shape_x - shape_margin_h,
                    top=shape_y - shape_margin_v,
                    right=shape_x + shape_w + shape_margin_h,
                    bottom=shape_y + shape_h - shape_margin_v)
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

                    # normalized_lip_array = (lip_array-[stable_mouth_crop_left, stable_mouth_crop_top]) * normalize_ratio

                    # get the crop mouth frame list when user is speaking
                    if speakingState == 0:
                        noSpeakingNormalizedCropMouthDeque.append(normalized_mouth_crop_image)
                        isSpeaking = False
                        speaking_length = 0
                    elif speakingState == 1:
                        speakingNormalizedCropMouthList = list(noSpeakingNormalizedCropMouthDeque)
                        speakingNormalizedCropMouthList.append(normalized_mouth_crop_image)
                        isSpeaking = True
                        speaking_length += 1

                    elif speakingState == 2:
                        speakingNormalizedCropMouthList.append(normalized_mouth_crop_image)
                        isSpeaking = True
                        speaking_length += 1

                        if speaking_length > 100:
                            # then reset the list
                            speakingNormalizedCropMouthList = []
                            noSpeakingNormalizedCropMouthDeque.clear()
                            speakingState = 0
                            speaking_length = 0
                            isSpeaking = False
                            force_detect_face = True

                    elif speakingState == 3:
                        # recognize what the user has spoken
                        # print(len(speakingNormalizedCropMouthList))
                        if len(speakingNormalizedCropMouthList) > gap_window_len + 10:  # a speech has to be over 1 second
                            if RECOGNIZER and USER_ADJUSTED and len(speakingNormalizedCropMouthList) > 15 and len(speakingNormalizedCropMouthList) < 100:
                                recognizeThread.setData(speakingNormalizedCropMouthList)
                        # then reset the list
                        speakingNormalizedCropMouthList = []
                        noSpeakingNormalizedCropMouthDeque.clear()
                        speakingState = 0
                        speaking_length = 0
                        isSpeaking = False


                #  Display the resulting frame
                # cv2.rectangle(image, (face_x, face_y), (face_x + face_w, face_y + face_h), (0, 255, 0), 2)
                # print("{} {} {} {}".format(stable_mouth_crop_left, stable_mouth_crop_top, stable_mouth_crop_width, stable_mouth_crop_height))
                if isSpeaking:
                    # cv2.rectangle(image, (mouth_x, mouth_y), (mouth_x + mouth_w, mouth_y + mouth_h), (255, 0, 0), 2)
                    cv2.rectangle(image, (stable_mouth_crop_left, stable_mouth_crop_top),
                                  (stable_mouth_crop_left + stable_mouth_crop_width,
                                   stable_mouth_crop_top + stable_mouth_crop_height), (0, 0, 255), 4)
                else:
                    # cv2.rectangle(image, (mouth_x, mouth_y), (mouth_x + mouth_w, mouth_y + mouth_h), (0, 255, 0), 2)
                    cv2.rectangle(image, (stable_mouth_crop_left, stable_mouth_crop_top),
                                  (stable_mouth_crop_left + stable_mouth_crop_width,
                                   stable_mouth_crop_top + stable_mouth_crop_height), (0, 255, 0), 4)
                for (x, y) in lip_array[12:20]:
                    cv2.circle(image, (x, y), 2, (255, 255, 255), -1)

                if recognizeThread.model_loaded and FPS_STARTED:
                    fps.update()

            cv2.imshow("frame", image)
            # Press Q on keyboard to stop
            # Press F on keyboard to force face re-detection on the next frame
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('f'):
                force_detect_face = True
                speakingNormalizedCropMouthList = []
                noSpeakingNormalizedCropMouthDeque.clear()
                speakingState = 0
                speaking_length = 0
                isSpeaking = False
            elif key == ord('u'):
                USER_ADJUSTED = True

    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    video_stream.stop()
    if RECOGNIZER:
        recognizeThread.stop()
    socketServer.stop()
    cv2.destroyAllWindows()

