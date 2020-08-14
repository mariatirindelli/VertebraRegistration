import os
import matplotlib.pyplot as plt

os.environ['PATH'] = 'C:\\Program Files\\ImFusion\\ImFusion Suite\\Suite;C:\\Program Files\\ImFusion' \
                     '\\ImFusion Suite\\Suite\\plugins;' + os.environ['PATH']
os.environ['PYTHONPATH'] = os.environ['PYTHONPATH'] + ';C:\\Program Files\\ImFusion\\ImFusion Suite\\Suite;'

import imfusion
imfusion.init()

from ImFusionPlugins import *


# plt.imshow(sp_segmentation)
# plt.show()

us_image = imfusion.open("E:\\Maria\\Submissions\\AnnaCollaboration\\ViennaDemo/us_label.imf")[0]
us_label = imfusion.open("E:\\Maria\\Submissions\\AnnaCollaboration\\ViennaDemo/centeredMRILable.imf")[0]
# matrix = us_image.matrix(0)

# us_image = imfusion.open("E:/tmpTrials/sweep1.imf")[0]
# us_label = imfusion.open("E:/tmpTrials/sweepLabels.imf")[0]

#imfusion.executeAlgorithm('Get Glob Center', [us_image, us_label])
# out  = imfusion.executeAlgorithm('Extract Label Frame', [us_image, us_label])[0]

imfusion.executeAlgorithm('Compute Rigid Translation', [us_label, us_image])