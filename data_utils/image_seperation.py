from data_utils.video_separation import fake_dir_name, real_dir_name, train_directory, get_project_root
import os
import cv2

skips = 4
fake_train_dir = os.path.join(get_project_root(), "data", "train", "fake_images")
real_train_dir = os.path.join(get_project_root(), "data", "train", "real_images")


def save_images(video_path: str, video_name: str, folder: str):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    while cap.isOpened():

        ret, frame = cap.read()
        if ret:
            if frame_count % skips == 0:
                cv2.imwrite(os.path.join(folder, video_name + "_" + str(frame_count) + ".jpg"), frame)
            frame_count = frame_count + 1
        else:
            break


if __name__ == "__main__":

    if not os.path.exists(fake_train_dir):
        os.mkdir(fake_train_dir)
    if not os.path.exists(real_train_dir):
        os.mkdir(real_train_dir)

    for sub_dir in os.listdir(train_directory):
        print(sub_dir)

        full_sub_dir = os.path.join(train_directory, sub_dir)
        fake_sub_dir = os.path.join(full_sub_dir, fake_dir_name)
        real_sub_dir = os.path.join(full_sub_dir, real_dir_name)

        for video in os.listdir(fake_sub_dir):

            video_path = os.path.join(fake_sub_dir, video)
            if os.path.isfile(video_path):
                save_images(video_path, video, fake_train_dir)

        for video in os.listdir(real_sub_dir):

            video_path = os.path.join(real_sub_dir, video)
            if os.path.isfile(video_path):
                save_images(video_path, video, real_train_dir)
