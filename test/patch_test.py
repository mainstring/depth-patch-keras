import cv2
import tensorflow as tf
import numpy as np

from data_utils.patch_generation import get_face, get_patches, PATCH_HEIGHT, PATCH_WIDTH
from train.patch_train import model_directory, classes

threshold = .6

if __name__ == "__main__":

    version = cv2.version
    print(version.opencv_version)
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
    config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

    cap = cv2.VideoCapture(0)
    frame_count = 0
    while cap.isOpened():

        cap.read()
        cap.read()
        ret, frame = cap.read()
        if ret:

            model = tf.keras.models.load_model(model_directory)
            face, w, h, x, y = get_face(frame)
            patches = get_patches(face, w, h)

            total_patches = len(patches)
            verdict_real = 0
            if total_patches > 0:
                patches = np.reshape(patches, [total_patches, PATCH_HEIGHT, PATCH_WIDTH, 3])
                patches = patches/255
                print(w, h)

                result = model.predict_on_batch(patches)
                print(result)
                result = np.argmax(result, axis=-1)
                print(result)
                # verdict = classes[result]
                verdict_real += np.sum(result)
                print(verdict_real)

                # for patch in patches:
                #     patch = np.reshape(patch, [1, PATCH_HEIGHT, PATCH_WIDTH, 3])
                #     patch = patch/255
                #     print(patch.shape)
                #
                #     result = model.predict_on_batch(patch)
                #     # result = np.argmax(result, axis=-1)
                #     print(result)

                verdict = "fake"
                if verdict_real/total_patches > threshold:
                    verdict = "real"
                cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
                cv2.putText(frame, verdict, (x,y), cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),1,cv2.LINE_AA)

            cv2.imshow("webcam", frame)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
