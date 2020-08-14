import os
import matplotlib.pyplot as plt
import numpy as np

os.environ['PATH'] = 'C:\\Program Files\\ImFusion\\ImFusion Suite\\Suite;C:\\Program Files\\ImFusion' \
                     '\\ImFusion Suite\\Suite\\plugins;' + os.environ['PATH']
os.environ['PYTHONPATH'] = os.environ['PYTHONPATH'] + ';C:\\Program Files\\ImFusion\\ImFusion Suite\\Suite;'

import imfusion
imfusion.init()

root = "E:\\Maria\\Submissions\\AnnaCollaboration\\MatthiasManual"
filename_list = ["sweep1", "Matthias_L5_L4"]

for filename in filename_list:

    filepath = os.path.join(root, filename) + ".imf"
    if not os.path.exists(filepath):
        print("path:  {} does not exists".format(filepath))
        continue
    shared_image = imfusion.open(filepath)[0]

    image_array = np.squeeze(np.array(shared_image))

    sum_array = np.sum(image_array, axis=(1, 2))
    min_idx = np.argmin(sum_array)

    to_plot = np.squeeze(image_array[min_idx, :, :])

    plt.imshow(to_plot)
    plt.show()
