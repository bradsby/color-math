from typing import List

import numpy as np

X_10 = 95.047
Y_10 = 100.000
Z_10 = 108.88


def xyz_to_cielab(data: List[float]) -> List[float]:
    x_data = data[0]
    y_data = data[1]
    z_data = data[2]

    var_x = x_data / X_10
    var_y = y_data / Y_10
    var_z = z_data / Z_10

    if var_x > 0.008856:
        var_x = var_x ** (1 / 3)
    else:
        var_x = (7.787 * var_x) + (16 / 116)
    if var_y > 0.008856:
        var_y = var_y ** (1 / 3)
    else:
        var_y = (7.787 * var_y) + (16 / 116)
    if var_z > 0.008856:
        var_z = var_z ** (1 / 3)
    else:
        var_z = (7.787 * var_z) + (16 / 116)

    L_val = (116 * var_y) - 16
    a_val = 500 * (var_x - var_y)
    b_val = 200 * (var_y - var_z)

    return [L_val, a_val, b_val]


def cielab_to_xyz(data: List[float]) -> List[float]:
    L_val = data[0]
    a_val = data[1]
    b_val = data[2]

    var_y = (L_val + 16) / 116
    var_x = a_val / 500 + var_y
    var_z = var_y - b_val / 200

    if var_y**3 > 0.008856:
        var_y = var_y**3
    else:
        var_y = (var_y - 16 / 116) / 7.787
    if var_x**3 > 0.008856:
        var_x = var_x**3
    else:
        var_x = (var_x - 16 / 116) / 7.787
    if var_z**3 > 0.008856:
        var_z = var_z**3
    else:
        var_z = (var_z - 16 / 116) / 7.787

    x_data = var_x * X_10
    y_data = var_y * Y_10
    z_data = var_z * Z_10

    return [x_data, y_data, z_data]


def xyz_to_hunterlab(data: List[float]) -> List[float]:
    x_data = data[0]
    y_data = data[1]
    z_data = data[2]

    var_ka = (175.0 / 198.04) * (Y_10 + X_10)
    var_kb = (70.0 / 218.11) * (Y_10 + Z_10)

    L_val = 100.0 * np.sqrt(y_data / Y_10)
    a_val = var_ka * (((x_data / X_10) - (y_data / Y_10)) / np.sqrt(y_data / Y_10))
    b_val = var_kb * (((y_data / Y_10) - (z_data / Z_10)) / np.sqrt(y_data / Y_10))

    return [L_val, a_val, b_val]


def hunterlab_to_xyz(data: List[float]) -> List[float]:
    L_val = data[0]
    a_val = data[1]
    b_val = data[2]

    var_ka = (175.0 / 198.04) * (Y_10 + X_10)
    var_kb = (70.0 / 218.11) * (Y_10 + Z_10)

    y_data = ((L_val / Y_10) ** 2) * 100.0
    x_data = (a_val / var_ka * np.sqrt(y_data / Y_10) + (y_data / Y_10)) * X_10
    z_data = -(b_val / var_kb * np.sqrt(y_data / Y_10) - (y_data / Y_10)) * Z_10

    return [x_data, y_data, z_data]


def rgb_to_xyz(data: List[float]) -> List[float]:
    s_r = data[0]
    s_g = data[1]
    s_b = data[2]

    var_r = s_r / 255
    var_g = s_g / 255
    var_b = s_b / 255

    if var_r > 0.04045:
        var_r = ((var_r + 0.055) / 1.055) ** 2.4
    else:
        var_r = var_r / 12.92
    if var_g > 0.04045:
        var_g = ((var_g + 0.055) / 1.055) ** 2.4
    else:
        var_g = var_g / 12.92
    if var_b > 0.04045:
        var_b = ((var_b + 0.055) / 1.055) ** 2.4
    else:
        var_b = var_b / 12.92

    var_r = var_r * 100
    var_g = var_g * 100
    var_b = var_b * 100

    x_data = var_r * 0.4124 + var_g * 0.3576 + var_b * 0.1805
    y_data = var_r * 0.2126 + var_g * 0.7152 + var_b * 0.0722
    z_data = var_r * 0.0193 + var_g * 0.1192 + var_b * 0.9505

    return [x_data, y_data, z_data]


def xyz_to_rgb(data: List[float]) -> List[float]:
    x_data = data[0]
    y_data = data[1]
    z_data = data[2]

    var_x = x_data / 100
    var_y = y_data / 100
    var_z = z_data / 100

    var_r = var_x * 3.2406 + var_y * -1.5372 + var_z * -0.4986
    var_g = var_x * -0.9689 + var_y * 1.8758 + var_z * 0.0415
    var_b = var_x * 0.0557 + var_y * -0.2040 + var_z * 1.0570

    if var_r > 0.0031308:
        var_r = 1.055 * (var_r ** (1 / 2.4)) - 0.055
    else:
        var_r = 12.92 * var_r
    if var_g > 0.0031308:
        var_g = 1.055 * (var_g ** (1 / 2.4)) - 0.055
    else:
        var_g = 12.92 * var_g
    if var_b > 0.0031308:
        var_b = 1.055 * (var_b ** (1 / 2.4)) - 0.055
    else:
        var_b = 12.92 * var_b

    s_r = var_r * 255
    s_g = var_g * 255
    s_b = var_b * 255

    return [s_r, s_g, s_b]


def cielab_to_hunterlab(data: List[float]) -> List[float]:
    return xyz_to_hunterlab(cielab_to_xyz(data))


def hunterlab_to_cielab(data: List[float]) -> List[float]:
    return xyz_to_cielab(hunterlab_to_xyz(data))


def rgb_to_hunterlab(data: List[float]) -> List[float]:
    return xyz_to_hunterlab(rgb_to_xyz(data))


def hunterlab_to_rgb(data: List[float]) -> List[float]:
    return xyz_to_rgb(hunterlab_to_xyz(data))
