import numpy as np


def conv_out_size(inp_size, kernel_size, dilation, padding, stride):
    return ((inp_size + 2*padding - dilation * (kernel_size - 1) - 1) // stride) + 1

def conv_trans_out_size(inp_size, kernel_size, dilation, padding, stride, out_padding):
    return (inp_size - 1) * stride - 2*padding + dilation * (kernel_size - 1) + out_padding + 1

def maxpool_size(inp_size, kernel_size=2, stride=0, padding=0, dilation=0):
  return ((inp_size + 2*padding - dilation * (kernel_size - 1) - 1) // stride) + 1


def extract_swath_observation(swath_map, cur_grid_y, win_width, win_height):
    
    swath = np.copy(swath_map)
    cropped_window = np.zeros((win_height, win_width))

    swath = swath[cur_grid_y:]
    swath_coords = np.argwhere(swath == 1)
    horizontal_mid = int(np.mean(swath_coords[:, 1]))

    # adjust horizontal crop
    x_low_map = horizontal_mid - win_width // 2
    x_high_map = horizontal_mid + win_width // 2
    if x_low_map < 0:
        x_gap = abs(x_low_map)
        x_low_map += x_gap
        x_high_map += x_gap

    elif x_high_map > swath_map.shape[1]:
        x_gap = x_high_map - swath_map.shape[1]
        x_low_map -= x_gap
        x_high_map -= x_gap
    
    x_low_win = 0
    x_high_win = win_width

    # adjust vertical crop
    y_low_map = cur_grid_y
    y_low_win = 0
    assert y_low_map >= 0, print("cropping y low bound negative! ", y_low_map)

    y_high_map = y_low_map + win_height
    y_high_win = win_height
    assert y_high_map <= swath_map.shape[0], print("y_high_map exceeds environment height", y_high_map)

    assert (y_high_map - y_low_map) == (y_high_win - y_low_win), print("y-dim not same size!", y_low_map, y_high_map, y_low_win, y_high_win)
    assert (x_high_map - x_low_map) == (x_high_win - x_low_win), print("x-dim not same size!", x_low_map, x_high_map, x_low_win, x_high_win)

    cropped_window[y_low_win:y_high_win, x_low_win:x_high_win] = swath_map[y_low_map:y_high_map, x_low_map:x_high_map]
    return cropped_window, x_low_map, x_high_map, y_low_map, y_high_map, x_low_win, x_high_win, y_low_win, y_high_win


def extract_observation_window(grid_map, win_width, win_height, x_low_map, x_high_map, y_low_map, y_high_map, x_low_win, x_high_win, y_low_win, y_high_win):
    
    # hardcoded handling of different map dimension, not great but should work
    assert len(grid_map.shape) <= 3, print("Invalid grid map dimension: ", grid_map.shape)
    
    if len(grid_map.shape) == 2:
        cropped_window = np.zeros((win_height, win_width))
    elif len(grid_map.shape) == 3:
        cropped_window = np.zeros((win_height, win_width, grid_map.shape[2]))

    cropped_window[y_low_win:y_high_win, x_low_win:x_high_win] = grid_map[y_low_map:y_high_map, x_low_map:x_high_map]
    return cropped_window
