import imfusion
import numpy as np
import os
import json


class LoadLabels(imfusion.Algorithm):
    def __init__(self, imageset):
        super().__init__()
        self.us_sweep = imageset

    @classmethod
    def convert_input(cls, data):

        print()
        # if len(data) != 1:
        #     raise imfusion.IncompatibleError('Requires only one inputs')

        return data

    def compute(self):

        labels = np.load("C:\\GitRepo\\VertebraRegistration\\tmp.npy")
        for i, sweep in enumerate(self.us_sweep):
            arr = np.array(self.us_sweep[i])
            toput = np.expand_dims(labels[i, :, :], -1)
            arr[toput >= 0.3] = 254
            arr[toput < 0.3] = 1
            self.us_sweep[i].assignArray(arr)
        print("DONE")

        return


imfusion.registerAlgorithm('Load Labels', LoadLabels)
