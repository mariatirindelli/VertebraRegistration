import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy.optimize import fmin_powell, fmin_l_bfgs_b

# image = np.array(Image.open("../tests/testImage1.png"))
# translated_image = image.copy()
#
# translated_image[250::, :] = image[0:250, :]
# translated_image[0:250, :] = 0
#
# plt.subplot(1, 2, 1)
# plt.imshow(image)
#
# plt.subplot(1, 2, 2)
# plt.imshow(translated_image)
# plt.show()
#
# save_image = Image.fromarray(translated_image)
# save_image.save("../tests/testImage1_translated.png")


def ind2sub(array_shape, ind):
    rows = (ind.astype('int') / array_shape[1])
    cols = (ind.astype('int') % array_shape[1]) # or numpy.mod(ind.astype('int'), array_shape[1])
    return (rows, cols)


def print_matrix(matrix):
    for i in range(matrix.shape[0]):
        print(matrix[i, :])


def align_center_of_mass(US_mask, US_transform, US_spacing, point_set):
    a = US_mask > 0
    US_index_points = a.nonzero()
    rows_index, cols_index = US_index_points

    xc_idx, yc_idx = np.mean(rows_index), np.mean(cols_index)

    A = np.eye()


def cost_function(rotz, deltax, deltay):
    profile_image = np.array(Image.open("../tests/testImage1.png").convert('L'))
    US_image = np.array(Image.open("../tests/testImage1_translated.png").convert('L'))
    a = profile_image > 0
    profile_points = a.nonzero()
    rows_index, cols_index = profile_points

    # defining the transformation matrix
    A = np.eye(3)
    A[1, 2] = deltay

    cost_function = 1
    for y, x in zip(rows_index, cols_index):
        pos_vector = np.array([x, y, 1])
        transformed_pos = np.matmul(A, pos_vector)

        row_index = int(transformed_pos[1])
        col_index = int(transformed_pos[0])

        if row_index >= US_image.shape[0] or col_index >= US_image.shape[1]:
            continue

        cost_function += US_image[row_index, col_index]

    return - cost_function

# values = []
# for i in range(0, 300):
#     values.append(cost_function(i))
#
# plt.plot(values)
# plt.show()



best_params = fmin_l_bfgs_b(cost_function, [100], approx_grad=True, bounds=[(220, 300)], epsilon=1e0)
print(best_params)