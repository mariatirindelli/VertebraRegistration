import imfusion
imfusion.init()

from ImFusionPlugins import *


us_image = imfusion.open("/home/maria/Desktop/singleUsImage.imf")[0]
output = imfusion.executeAlgorithm('Get Glob Center', [us_image])[0]