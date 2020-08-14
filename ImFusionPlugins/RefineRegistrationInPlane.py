import imfusion
import numpy as np
import os
import json


class ApplyRigidAlignment(imfusion.Algorithm):
    def __init__(self, mri_label, us_label):
        super().__init__()
        self.mri_label = mri_label[0]
        self.us_label = us_label[0]
        self.imageset_out = imfusion.SharedImageSet()

        self.json_path = "C:\\GitRepo\\VertebraRegistration\\transform.json"

    @classmethod
    def convert_input(cls, data):

        if len(data) != 2:
            raise imfusion.IncompatibleError('Requires two inputs')

        if not isinstance(data[0], imfusion.SharedImageSet) or data[0].img().dimension() != 3 \
                or data[0].modality != imfusion.Data.Modality.LABEL:
            raise imfusion.IncompatibleError('First input must be a 3D label map')

        if not isinstance(data[1], imfusion.SharedImageSet) or data[1].img().dimension() != 2 \
                or data[1].modality != imfusion.Data.Modality.LABEL:
            raise imfusion.IncompatibleError('Second input must be a 2D label map')

        return data

    def compute(self):
        # clear output of previous runs
        self.imageset_out.clear()

        return


imfusion.registerAlgorithm('Apply Rigid Alignment', ApplyRigidAlignment)
