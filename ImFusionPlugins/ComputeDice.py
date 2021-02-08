import imfusion
import numpy as np


class ComputeDice(imfusion.Algorithm):
    def __init__(self, labelmap, gt):
        super().__init__()
        self.labelmap = labelmap[0]
        self.gt = gt[0]

    @classmethod
    def convert_input(cls, data):

        if len(data) != 2:
            raise imfusion.IncompatibleError('Requires two inputs')

        if not isinstance(data[0], imfusion.SharedImageSet) or data[0].img().dimension() != 3:
            raise imfusion.IncompatibleError('First input must be a 3D label map')

        if not isinstance(data[1], imfusion.SharedImageSet) or data[1].img().dimension() != 3:
            raise imfusion.IncompatibleError('Second input must be a 3D label map')

        return data

    def compute(self):
        labelmap_array = np.squeeze(np.array(self.labelmap))
        gt_array = np.squeeze(np.array(self.gt))

        dice = np.sum(labelmap_array[gt_array == 1]) * 2.0 / (np.sum(labelmap_array) + np.sum(gt_array))

        print('Dice similarity score is {}'.format(dice))


imfusion.registerAlgorithm('Compute Dice', ComputeDice)

