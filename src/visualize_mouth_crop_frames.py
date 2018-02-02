import pprint, pickle
import cv2
from time import sleep

ID = 'gyz_flip-42-frame_8_image.pkl'
subject, gestureId, filename = ID.split("-")
file_path = "../resource/data_new_augmented_array/" + subject + "/" + str(gestureId) + "/" + filename

image_pkl_file = open(file_path, 'rb')
mouth_image_list = pickle.load(image_pkl_file)
image_pkl_file.close()


cv2.namedWindow("frame");
cv2.moveWindow("frame", 1000, 400);

print(len(mouth_image_list))

i = 0
while True:
    image = mouth_image_list[i][:]
    cv2.imshow('frame', image)
    key = cv2.waitKey(30) & 0xFF
    i = min(len(mouth_image_list) - 1, i + 1)

    if key == ord('q'):
        break

cv2.destroyAllWindows()
