from pathlib import Path
import os

fake_dir_name = "fake"
real_dir_name = "real"
fake_files = ['3.avi', '4.avi', '5.avi', '6.avi', '7.avi', '8.avi', 'HR_2.avi', 'HR_3.avi']
real_files = ['1.avi', '2.avi', 'HR_1.avi', 'HR_4.avi']


def get_project_root() -> Path:
    return Path(__file__).parent.parent


print(get_project_root())
test_directory = os.path.join(get_project_root(), "data", "CASIA_faceAntisp", "test_release")
train_directory = os.path.join(get_project_root(), "data", "CASIA_faceAntisp", "train_release")
print(os.listdir(train_directory))


if __name__ == "__main__":
    for sub_dir in os.listdir(train_directory):
        print(sub_dir)

        full_sub_dir = os.path.join(train_directory, sub_dir)
        fake_sub_dir = os.path.join(full_sub_dir, fake_dir_name)
        real_sub_dir = os.path.join(full_sub_dir, real_dir_name)

        if not os.path.exists(fake_sub_dir):
            os.mkdir(fake_sub_dir)

            for fake_file in fake_files:
                try:
                    Path(os.path.join(full_sub_dir, fake_file)).rename(os.path.join(fake_sub_dir, fake_file))
                except Exception as e:
                    print(e)
                    print("error for: " + os.path.join(full_sub_dir, fake_file))

        if not os.path.exists(real_sub_dir):
            os.mkdir(real_sub_dir)

            for real_file in real_files:
                try:
                    Path(os.path.join(full_sub_dir, real_file)).rename(os.path.join(real_sub_dir, real_file))
                except Exception as e:
                    print(e)
                    print("error for: " + os.path.join(full_sub_dir, real_file))
