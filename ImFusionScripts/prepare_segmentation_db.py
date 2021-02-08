import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import shutil
from imfusion_import import *
import logging
from generate_db_batch import prepare_data_list

imfusion_exported_data_path = "D:\\Maria\\DataBases\\SpineIFL\\BoneSemegmentationDb\\exported_data"
save_path = "D:\\NAS\\BoneSegmentation"
test_subjects = ["Maria",
                 "MariaT_0",
                 "MariaT_1",
                 "MariaT_2",
                 "MariaT_3",
                 "MariaT_4",
                 "MariaT_5",
                 "MariaT_6",
                 "MatthiasS"]

cross_val_folders = 5


def get_subject_id(name, subject_dict):
    if "MariaT" in name:
        name = "MariaT"

    if name not in subject_dict.keys():
        subject_dict["currentId"] += 1
        subject_dict[name] = str(subject_dict["currentId"])
    return subject_dict[name], subject_dict


def save_db(cross_val_folder, logger):

    # Prepare train, val, test folder if not present
    print(cross_val_folder)
    if os.path.exists(os.path.join(save_path, cross_val_folder)):
        ans = input("Do you want to overwrite existing folder?")
        if ans == 'N' or ans == 'n':
            return
        shutil.rmtree(os.path.join(save_path, cross_val_folder))

    os.mkdir(os.path.join(save_path, cross_val_folder))
    for split in ['train', 'val', 'test']:
        os.mkdir(os.path.join(save_path, cross_val_folder, split))
        os.mkdir(os.path.join(save_path, cross_val_folder, split, 'images'))
        os.mkdir(os.path.join(save_path, cross_val_folder, split, 'labels'))

    fh = logging.FileHandler(os.path.join(save_path, cross_val_folder, "log_info.log"))
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    # Get the data list with correct split between subjects
    # it returns a list as [[input_image_path1, input_label_path1, split1],
    # [input_image_path2, input_label_path2, split2], ...]
    # example:
    # [['../BoneSemegmentationDb/exported_data/0.imf', '../BoneSemegmentationDb/exported_data/0-label.imf', 'test']
    # ['../BoneSemegmentationDb/exported_data/1.imf', '../BoneSemegmentationDb/exported_data/1-label.imf', 'train']
    # [..]]

    data_list = prepare_data_list(input_db_path=imfusion_exported_data_path,
                                  output_save_path=save_path,
                                  val_percentage=0.2,
                                  test_list=test_subjects,
                                  logger=logger)

    subject_dict = {"currentId":0}

    for (sweep_path, label_path, split, subject_name) in data_list:

        if not os.path.exists(sweep_path) or not os.path.exists(label_path):
            print("File {} or Label: {} does not exits".format(sweep_path, label_path))
            continue
        subject_id, subject_dict = get_subject_id(subject_name, subject_dict)

        sweep = imfusion.open(sweep_path)[0]
        labels = imfusion.open(label_path)[0]
        sweeep_id = os.path.split(sweep_path)[-1].strip(".imf")

        iterator = 0
        for image, label in zip(sweep, labels):

            image_array = np.array(image)
            label_array = np.array(label)

            label_array[label_array == 1] = 0
            label_array[label_array == 3] = 1

            if np.sum(label_array) == 0:
                continue

            image_save_path = os.path.join(save_path, cross_val_folder, split, "images", subject_id + "_" + sweeep_id + "_" + str(iterator) + ".png")
            label_save_path = os.path.join(save_path, cross_val_folder, split, "labels",  subject_id + "_" + sweeep_id + "_" + str(iterator) + "_label.png")

            save_png(np.squeeze(image_array), image_save_path)
            save_png(np.squeeze(label_array), label_save_path)

            iterator += 1
    for key in subject_dict:
        logger.info("subject name: {} - subject id: {}".format(key, subject_dict[key]))

    logger.handlers.clear()


def save_png(data_array, path):
    im = Image.fromarray(data_array)
    im = im.transpose(Image.FLIP_TOP_BOTTOM )
    im.save(path)


def show():
    image_ids = [item.split(".")[0] for item in os.listdir(os.path.join(save_path, "images"))]

    for image_id in image_ids:
        image_path = os.path.join(save_path, "images", image_id + ".png")
        label_path = os.path.join(save_path, "labels", image_id + "_label.png")

        image = plt.imread(image_path)
        label = plt.imread(label_path)

        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.subplot(1, 2, 2)
        plt.imshow(label * 255)

        plt.show()


if __name__ == '__main__':
    logger = logging.getLogger('prepare_segmentation')
    logger.setLevel(logging.INFO)
    for i in range(cross_val_folders):
        save_db("cross_val" + str(i+1), logger)
