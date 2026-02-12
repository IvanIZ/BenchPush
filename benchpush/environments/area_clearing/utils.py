import numpy as np

# Helper functions
def round_up_to_even(x):
    return int(np.ceil(x / 2) * 2)

def position_to_pixel_indices(x, y, image_shape, local_map_pixels_per_meter):
    pixel_i = np.floor(image_shape[0] / 2 - y * local_map_pixels_per_meter).astype(np.int32)
    pixel_j = np.floor(image_shape[1] / 2 + x * local_map_pixels_per_meter).astype(np.int32)
    pixel_i = np.clip(pixel_i, 0, image_shape[0] - 1)
    pixel_j = np.clip(pixel_j, 0, image_shape[1] - 1)
    return pixel_i, pixel_j

def pixel_indices_to_position(pixel_i, pixel_j, image_shape, local_map_pixels_per_meter):
    x = (pixel_j - image_shape[1] / 2) / local_map_pixels_per_meter
    y = (image_shape[0] / 2 - pixel_i) / local_map_pixels_per_meter
    return x, y