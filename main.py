import os
import numpy as np

os.environ['PATH'] = 'C:\\Program Files\\ImFusion\\ImFusion Suite\\Suite;C:\\Program Files\\ImFusion' \
                     '\\ImFusion Suite\\Suite\\plugins;' + os.environ['PATH']
os.environ['PYTHONPATH'] = os.environ['PYTHONPATH'] + ';C:\\Program Files\\ImFusion\\ImFusion Suite\\Suite;'

import imfusion
imfusion.init()


file_dir = "E:\\Maria\\Submissions\\IROS2020\\Experiments\\IROSDb\\dataFullImageOnly"
file_list = os.listdir(file_dir)
file_list = [item for item in file_list if ".imf" in item]

for item in file_list:
    file_data = os.path.join("E:\\Maria\\Submissions\\IROS2020\\Experiments\\IROSDb\\data\\" + item)
    file_full_img = os.path.join("E:\\Maria\\Submissions\\IROS2020\\Experiments\\IROSDb\\dataFullImageOnly\\" + item)

    if not os.path.exists(file_data):
        print(file_data)
        continue

    data_sweep = imfusion.open(file_data)[0]
    data_full_sweep = imfusion.open(file_full_img)[0]

    diff_in_data = False
    diff_in_labels = False
    diff_in_csv = False

    for image_data, image_full in zip(data_sweep, data_full_sweep):
        array_data = np.array(image_data)
        array_full = np.array(image_full)

        if np.sum(np.abs(array_data - array_full)) != 0:
            print("data: {} differs".format(item))
            diff_in_data = True

    label_sweep = imfusion.open(file_data)[1]
    label_full_sweep = imfusion.open(file_full_img)[1]

    for label_data, label_full in zip(label_sweep, label_full_sweep):
        array_data_label = np.array(label_data)
        array_full_label = np.array(label_full)

        if np.sum(np.abs(array_data_label - array_full_label)) != 0:
            print("label: {} differs".format(item))
            diff_in_labels = True


    # comparing csv
    csv_data_path = os.path.join("E:\\Maria\\Submissions\\IROS2020\\Experiments\\IROSDb\\data\\" + item[0:-4] + ".csv")
    csv_full_path = os.path.join("E:\\Maria\\Submissions\\IROS2020\\Experiments\\IROSDb\\dataFullImageOnly\\" + item[0:-4] + ".csv")
    with open(csv_data_path, 'r') as fid:
        csv_data = fid.readlines()

    with open(csv_full_path, 'r') as fid:
        csv_full = fid.readlines()

    for line_data, line_full in zip(csv_data, csv_full):
        if line_data != line_full:
            diff_in_csv = True


    print("processed item: {} -- difference in data: {} -- difference in labels: {} -- difference in csv: {}".format(
        item, diff_in_data, diff_in_labels, diff_in_csv))




from ImFusionPlugins import *


# plt.imshow(sp_segmentation)
# plt.show()

# us_image = imfusion.open("E:\\Maria\\Submissions\\AnnaCollaboration\\ViennaDemo/us_label.imf")[0]
# us_label = imfusion.open("E:\\Maria\\Submissions\\AnnaCollaboration\\ViennaDemo/centeredMRILable.imf")[0]
# # matrix = us_image.matrix(0)
#
# # us_image = imfusion.open("E:/tmpTrials/sweep1.imf")[0]
# # us_label = imfusion.open("E:/tmpTrials/sweepLabels.imf")[0]
#
# #imfusion.executeAlgorithm('Get Glob Center', [us_image, us_label])
# # out  = imfusion.executeAlgorithm('Extract Label Frame', [us_image, us_label])[0]
#
# imfusion.executeAlgorithm('Compute Rigid Translation', [us_label, us_image])


# us_image = imfusion.open("E:\\Maria\\DataBases\\SpineIFL\\BoneSemegmentationDb\\labels\\1.imf")[0]
# us_label = imfusion.open("E:\\Maria\\DataBases\\SpineIFL\\BoneSemegmentationDb\\labels\\1-labels.imf")[0]
# imfusion.executeAlgorithm('Transfer Transfer Stream', [us_label, us_image])

# labelmap = imfusion.open("C:\\Users\\maria\\OneDrive\\Desktop\\FaridVolumes\\5F1670_real_seg.mhd")[0]
# gt = imfusion.open("C:\\Users\\maria\\OneDrive\\Desktop\\FaridVolumes\\mr_tree.mhd")[0]
# imfusion.executeAlgorithm('Compute Dice', [labelmap, gt])
#



