import imfusion
import numpy as np
import SimpleITK as sitk

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

    def fun(self, x, pc_us, CT, T_v0):
        # get vertebra point clouds

        assert isinstance(CT, sitk.Image)

        CT_array = sitk.GetArrayFromImage(CT)
        a = np.argwhere(CT_array == 1)

        V1 = [CT.TransformIndexToPhysicalPoint(item.tolist()) for item in np.argwhere(CT_array == 1)]
        V2 = [CT.TransformIndexToPhysicalPoint(item.tolist()) for item in np.argwhere(CT_array == 2)]

        V1 = np.array(V1)
        V2 = np.array(V2)

        # assume pure translation for now
        T1 = np.eye(1)
        T1[0, 3] = x[0]

        T2 = np.eye(1)
        T2[0, 3] = x[1]

        V1_t = T1 * V1
        V2_t = T2 * V2

        seedImage = sitk.Image(CT.GetSize()[0], CT.GetSize()[1], sitk.sitkUInt8)
        seedImage.SetSpacing(CT.GetSpacing())
        seedImage.SetOrigin(CT.GetOrigin())
        seedImage.SetDirection(CT.GetDirection())

        seedImage[seedImage.TransformPhysicalPointToIndex(V1_t)] = 1
        seedImage[seedImage.TransformPhysicalPointToIndex(V2_t)] = 1

        # compute dice with US
        dice_vi = 2*np.sum()

        # compute regularization loss between vertebra

        pass


    def compute(self):

        # the point cloud for the us contains x, y, z, probability_of_bones
        point_cloud_us = None

        point_cloud_v1 = None
        point_cloud_v2 = None

        # the transformation optimization for a single vertebra
        pass


imfusion.registerAlgorithm('Compute Dice', ComputeDice)

