import imfusion
import numpy as np


class ExtractLabelFrame(imfusion.Algorithm):
    def __init__(self, us_sweep, us_label):
        super().__init__()
        self.us_sweep = us_sweep
        self.us_label = us_label
        self.imageset_out = imfusion.SharedImageSet()

    @classmethod
    def convert_input(cls, data):

        if len(data) != 2:
            raise imfusion.IncompatibleError('Requires two inputs')

        us_sweep, us_label = data

        # TODO: run all checks on data

        # if not isinstance(image, imfusion.SharedImageSet) or image.img().dimension() != 3:
        #     raise imfusion.IncompatibleError('First input must be a volume')
        #
        # if not isinstance(mask, imfusion.SharedImageSet) or mask.modality != imfusion.Data.Modality.LABEL:
        #     raise imfusion.IncompatibleError('Second input must be a label map')
        return {'us_sweep': us_sweep, 'us_label': us_label}

    def compute(self):
        # clear output of previous runs
        self.imageset_out.clear()

        sp_idx = -1
        for i, img in enumerate(self.us_label):
            if np.sum(img) > 0:
                sp_idx = i

        if sp_idx < 0:
            return

        us_image_label_array = np.array(self.us_label[sp_idx])

        output_image = imfusion.SharedImage(us_image_label_array)

        # self.us_sweep.matrix(sp_idx) gives me the data to world transformation, while the image matrix should
        # be the world to data registration and therefore it has to be inverted beforehand
        output_image.matrix = np.linalg.inv(self.us_sweep.matrix(sp_idx).copy())
        output_image.spacing = self.us_sweep[sp_idx].spacing.copy()

        self.imageset_out.add(output_image)
        print("")

    def output(self):
        return [self.imageset_out]


imfusion.registerAlgorithm('Extract Label Frame', ExtractLabelFrame)
