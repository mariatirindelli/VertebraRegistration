import imfusion
import numpy as np
import os
import json


class ApplyRigidAlignment(imfusion.Algorithm):
    def __init__(self, imageset):
        super().__init__()
        self.imageset = imageset
        self.imageset_out = imfusion.SharedImageSet()
        self.json_path = "C:\\GitRepo\\VertebraRegistration\\transform.json"

    @classmethod
    def convert_input(cls, data):

        if len(data) != 1:
            raise imfusion.IncompatibleError('Requires only one inputs')

        return data

    def compute(self):
        # clear output of previous runs
        self.imageset_out.clear()

        # load the transformation matrix from the json
        if os.path.exists(self.json_path):
            with open(self.json_path, 'r') as fp:
                transform_dict = json.load(fp)

        else:
            return

        if "alignmentT" not in transform_dict.keys():
            return

        T_alignment = np.array(transform_dict["alignmentT"])

        # get the image data to world transformation matrix
        T = np.linalg.inv(self.imageset[0].matrix)

        # compute the new data to world transformation matrix to align the input image
        T_new = np.matmul(T_alignment, T)

        # set the image matrix. Since by convention the image matrix is from world to data, we need to take the inverse
        self.imageset[0].matrix = np.linalg.inv(T_new)

        return


imfusion.registerAlgorithm('Apply Rigid Alignment', ApplyRigidAlignment)