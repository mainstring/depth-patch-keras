from data_utils.image_seperation import fake_train_dir, real_train_dir
from data_utils.video_separation import get_project_root
import os
import cv2
import random

fake_patch_directory = os.path.join(get_project_root(), "data", "train", "patch", "fake")
real_patch_directory = os.path.join(get_project_root(), "data", "train", "patch", "real")

# Load the cascade
# classifier_cuda = cuda_CascadeClassifier('cascades_file.xml')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

MAX_PATCH_PER_FACE = 20
PATCH_WIDTH = 96
PATCH_HEIGHT = 96


def get_face(img):
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw rectangle around the faces
    if len(faces) > 0:
        x, y, w, h = faces[0]
        return img[y:y + h, x:x + w], w, h, x, y
    return None, 0, 0, 0, 0


def get_patches(face, w, h):
    patches = []
    if face is not None and w > 100 and h > 100:

        total_patches = min(int(w / PATCH_WIDTH)*2, int(h / PATCH_HEIGHT)*2, MAX_PATCH_PER_FACE)

        for i in range(total_patches):
            x = random.randint(0, w - PATCH_WIDTH - 1)
            y = random.randint(0, h - PATCH_HEIGHT - 1)
            patch = face[y:y + PATCH_HEIGHT, x:x + PATCH_WIDTH]
            patches.append(patch)
    return patches


def save_patches(face, w, h, folder):

    patches = get_patches(face, w, h)
    for patch in patches:
        file_count = len(os.listdir(folder))
        name = str(file_count + 1) + ".jpg"
        cv2.imwrite(os.path.join(folder, name), patch)


if __name__ == "__main__":

    if not os.path.exists(fake_patch_directory):
        os.mkdir(fake_patch_directory)
    if not os.path.exists(real_patch_directory):
        os.mkdir(real_patch_directory)

    for img_path in os.listdir(fake_train_dir):

        file = os.path.join(fake_train_dir, img_path)
        if os.path.isfile(file):
            # Read the input image
            img = cv2.imread(file)
            face, w, h = get_face(img)
            save_patches(face, w, h, fake_patch_directory)

    for img_path in os.listdir(real_train_dir):

        file = os.path.join(real_train_dir, img_path)
        if os.path.isfile(file):
            # Read the input image
            img = cv2.imread(file)
            face, w, h = get_face(img)
            save_patches(face, w, h, real_patch_directory)