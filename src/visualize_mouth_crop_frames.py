import pprint, pickle
import cv2

outputFolder = "../resource/data/sk/1/"
outputFileIndex = 6
image_pkl_file = open(outputFolder + "frame_" + str(outputFileIndex) + "_image.pkl", 'rb')
mouth_image_list = pickle.load(image_pkl_file)
image_pkl_file.close()

lip_pkl_file = open(outputFolder + "frame_" + str(outputFileIndex) + "_lip.pkl", 'rb')
mouth_lip_list = pickle.load(lip_pkl_file)
lip_pkl_file.close()

cv2.namedWindow("frame");
cv2.moveWindow("frame", 1000, 400);

print(len(mouth_image_list))
print(len(mouth_lip_list))

i = 0
while True:
    image = mouth_image_list[i][:]
    lip = mouth_lip_list[i]

    for (x, y) in lip:
        cv2.circle(image, (int(x), int(y)), 1, (255, 255, 255), -1)

    cv2.imshow('frame', image)

    key = cv2.waitKey(0) & 0xFF
    if key == ord('j'):
        i = max(0, i-1)
    elif key == ord('k'):
        i = min(len(mouth_image_list)-1, i+1)
    elif key == ord('q'):
        break

cv2.destroyAllWindows()
