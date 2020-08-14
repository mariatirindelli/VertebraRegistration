import imfusion
import numpy as np
import json
import os


class ComputeRigidTranslation(imfusion.Algorithm):
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

    @staticmethod
    def get_centroid(shared_img):

        # get image spacing:
        spacing_x = shared_img.spacing[0]
        spacing_y = shared_img.spacing[1]
        if len(shared_img.spacing) == 2:
            spacing_z = 1
        else:
            spacing_z = shared_img.spacing[2]

        # get image transformation matrix. The function shared_img.matrix provide the transformation from world
        # coordinate to image coordinate. Therefore to obtain the transfomation from data coordinate to world
        # coordinate we need to take the inverse.
        Tdata2world = np.linalg.inv(shared_img.matrix)
        print(Tdata2world)

        # Convert the label image into an array
        img_array = np.array(shared_img)

        # remove last singleton dimension (channels). There is anyway only one channels for all input images
        img_array = np.squeeze(img_array)

        # If the input shared_img is the volumes, the array has shape: [slices x height x width].
        # Therefore it has to be permuted to be have shape: [height x width x slices]
        if len(img_array.shape) == 3:
            img_array = np.transpose(img_array, [1, 2, 0])

        # If the input shared_img is an ultrasound image, add a dimension representing the image slices to make it
        # compatible with the 3D mri labels. The ultrasound image label will have shape:
        # [height x width x slices] = [height x width x 1]
        if len(img_array.shape) == 2:
            img_array = np.expand_dims(img_array, axis=-1)

        # get center coordinates in the pixel space
        row_o = int(img_array.shape[0]/2)
        col_o = int(img_array.shape[1]/2)
        depth_o = int(img_array.shape[2]/2)

        # find the glob centroid expressed in pixels
        row_nz, col_nz, chan_nz = np.nonzero(img_array)
        row_c = np.mean(row_nz)  # center row index
        col_c = np.mean(col_nz)  # center col index
        depth_c = np.mean(chan_nz)  # center depth index

        # get the glob centroid expressed wrt to the image reference frame
        x_c = (col_c - col_o)*spacing_x
        y_c = (row_c - row_o) * spacing_y
        z_c = (depth_c - depth_o)*spacing_z
        im_C = np.array([x_c, y_c, z_c, 1])

        # get the glob centroid expressed in the world reference frame
        w_C = np.matmul(Tdata2world, im_C)

        return w_C

    @staticmethod
    def compute_translation(c_mri, c_us):
        T = np.eye(4)
        translation = c_us - c_mri

        T[:, 3] = translation
        T[3, 3] = 1
        return T

    def update_json(self, T):

        if os.path.exists(self.json_path):
            with open(self.json_path, 'r') as fp:
                transform_dict = json.load(fp)

        else:
            transform_dict = dict()

        transform_dict["alignmentT"] = T.tolist()
        print(transform_dict)

        with open(self.json_path, 'w') as fp:
            json.dump(transform_dict, fp)

    def compute(self):
        # clear output of previous runs
        self.imageset_out.clear()

        # get the ultrasound labelmap centroid in the world reference frame
        c_us = self.get_centroid(self.us_label)

        # get the mri labelmap centroid in the world reference frame
        c_mri = self.get_centroid(self.mri_label)

        # get the translation matrix to align the mri centroid on the us centroid
        T_mri_alignment = self.compute_translation(c_mri=c_mri, c_us=c_us)

        # update the json file where the alignment transformation is saved. This is the transformation to apply to the
        # MRI transformation matrix (data to world matrix) to align its sp centroid with the ultrasound one
        self.update_json(T_mri_alignment)

        # add images centroid as annotation labels in the imFusion gui
        am = imfusion.app.annotationModel()

        for name, point in zip(["Label US", "Label MRI"], [c_us, c_mri]):
            annotation = am.createAnnotation(imfusion.AnnotationType.Point)
            annotation.color = (1.0, 0.0, 0.0)  # make it red
            annotation.points = [(point[0], point[1], point[2])]
            annotation.name = name


imfusion.registerAlgorithm('Compute Rigid Translation', ComputeRigidTranslation)
