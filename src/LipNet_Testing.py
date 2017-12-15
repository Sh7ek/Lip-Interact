from keras.models import load_model
import pickle
import numpy as np
import datetime

model = load_model('../resource/2017-12-07_model_sk_out.h5')

validation_filename = "../resource/testing_list_sk_out.txt"
with open(validation_filename) as f:
    list_validation_IDs = f.readlines()
    list_validation_IDs = [x.strip() for x in list_validation_IDs]

total_seconds = 0.0
total_instances = 0
right_instances = 0
# start = datetime.datetime.now()
for ID in list_validation_IDs:
    subject, gestureId, filename = ID.split("-")
    file_path = "../resource/data_augmented_array/" + subject + "/" + str(gestureId) + "/" + filename
    image_pkl_file = open(file_path, 'rb')
    mouth_image_array = pickle.load(image_pkl_file).astype(dtype=np.float32)  # frame_n, img_h, img_w, img_c
    image_pkl_file.close()

    start = datetime.datetime.now()

    frame_n_temp = min(mouth_image_array.shape[0], 70)
    X = np.zeros((1, 70, 80, 100, 3), dtype=np.float32)
    X[0, 0:frame_n_temp] = mouth_image_array[0:frame_n_temp]

    y = int(gestureId)
    if y > 6:
        y -= 1

    y_predict_probabilities = model.predict(X, batch_size=None)[0]
    y_predict = y_predict_probabilities.argmax() + 1

    if y_predict == y:
        right_instances += 1
    else:
        print("{} is recognized as {}".format(ID, y_predict))
        print(y_predict_probabilities)
    total_instances += 1

    if total_instances % 100 == 0:
        print(total_instances)

    end = datetime.datetime.now()
    total_seconds += (end - start).total_seconds()

# end = datetime.datetime.now()
# total_seconds = (end - start).total_seconds()

print("time: {}   instances: {}   speed: {} s".format(total_seconds, len(list_validation_IDs), total_seconds/len(list_validation_IDs)))

accuracy = float(right_instances)/total_instances
print("accuracy: {}".format(accuracy))
