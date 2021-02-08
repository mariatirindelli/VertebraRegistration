import os
import random
import pandas as pd
import logging

db_path = "D:\\Maria\\DataBases\\SpineIFL\\BoneSemegmentationDb\\exported_data"
output_path = "D:\\NAS\\BoneSegmentation"

test_subjects = ["Maria", "MariaT_0", "MariaT_1", "MariaT_2", "MariaT_3", "MariaT_4", "MariaT_5", "MariaT_6",
                 "MatthiasS"]
val_perc = 0.2


def prepare_data_list(input_db_path, output_save_path, val_percentage, test_list, logger):

    assert 0 <= val_percentage <= 1, "Validation percentage must be between 0 and 1"

    # 1. Read data_list and extract the list of subject names as [Ardit.imf, Luigi.img, Mario.imf]
    data_list_path = os.path.join(input_db_path, "data_list.txt")
    pd_frame = pd.read_csv(data_list_path, sep="\t")
    subject_list = [os.path.split(item)[-1] for item in pd_frame["originalDataPath"]]
    subjects_list = list(set(subject_list))

    # 2. Removing test subjects
    test_list = [item + ".imf" for item in test_list if ".imf" not in item]
    subject_list = [item for item in subjects_list if item not in test_list]

    # 3. Splitting between train and validation
    num_validation_subjects = int(val_percentage*len(subject_list))
    val_list = random.sample(population=subject_list,
                             k=num_validation_subjects)
    train_list = [item for item in subject_list if item not in val_list]

    logger.info("train list: " + str(train_list))
    logger.info("val list: " + str(val_list))
    logger.info("test list: " + str(test_list))

    batch_lines = []
    return_list = []
    for i in range(len(pd_frame.index)):

        input_image = os.path.join(input_db_path, pd_frame['#dataPath'][i])
        input_label = os.path.join(input_db_path, pd_frame['labelPath'][i])
        subject_name = os.path.split(pd_frame['originalDataPath'][i])[-1]

        if subject_name in train_list:
            output_image_path = os.path.join(output_save_path, "train", "images")
            output_label_path = os.path.join(output_save_path, "train", "labels")
            split = "train"

        elif subject_name in val_list:
            output_image_path = os.path.join(output_save_path, "val", "images")
            output_label_path = os.path.join(output_save_path, "val", "labels")
            split = "val"

        elif subject_name in test_list:
            output_image_path = os.path.join(output_save_path, "test", "images")
            output_label_path = os.path.join(output_save_path, "test", "labels")
            split = "test"

        else:
            print(subject_name)
            continue
        batch_lines.append("\n" + str(i) + ";" + str(i) + "_label;" + input_image + ";" + input_label + ";" +
                           output_image_path + ";" + output_label_path)
        return_list.append([input_image, input_label, split, subject_name])

    with open(os.path.join(output_save_path, "batch_path.txt"), 'w') as fid:
        fid.write("NAMEIMAGE;NAMELABEL;INPUTIMAGE;INPUTLABEL;OUTPUTIMAGE;OUTPUTLABEL")
        fid.writelines(batch_lines)

    return return_list


if __name__ == '__main__':
    logger = logging.getLogger('spam_application')
    prepare_data_list(db_path, output_path, val_perc, test_subjects, logger)

