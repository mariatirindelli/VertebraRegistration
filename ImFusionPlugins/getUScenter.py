import imfusion
import numpy as np


class SPSegmentation(imfusion.Algorithm):
    def __init__(self, imageset):
        super().__init__()
        self.imageset = imageset
        self.imageset_out = imfusion.SharedImageSet()

    @classmethod
    def convert_input(cls, data):
        if len(data) == 1 and isinstance(data[0], imfusion.SharedImageSet):
            return data
        raise RuntimeError('Requires exactly one image')

    def compute(self):
        # clear output of previous runs
        self.imageset_out.clear()

        # take the first (and in theory only) ultrasound image from the input imageset
        us_image = self.imageset[0]

        # TODO: replace with automatic method for ultrasound segmentation
        # here it should compute the Spinous Process segmentation in the ultrasound image. For now it only loads the
        # segmentation from a file
        sp_segmentation = imfusion.open("/media/maria/Elements1/tmpTrials/segmentedVertebra.imf")
        sp_segmentation = np.array(sp_segmentation[0])

        # find the glob centroid expressed in pixels
        r_nz, c_nz = np.nonzero(sp_segmentation)
        r_c = np.mean(r_nz) # center row index
        c_c = np.mean(c_nz) # center col index

        x_o = 0
        y_o = 0


        x_c = c_c


        # compute the thresholding on each individual image in the set
        for image in self.imageset:
            arr = np.array(image) # creates a copy
            arr[arr < 2500] = 0
            arr[arr >= 2500] = 1

            # create the output image from the thresholded data
            image_out = imfusion.SharedImage(arr)
            self.imageset_out.add(image_out)

    def output(self):
        return [self.imageset_out]


imfusion.registerAlgorithm('Get Glob Center', SPSegmentation)
