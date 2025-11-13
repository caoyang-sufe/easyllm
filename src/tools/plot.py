# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@stu.sufe.edu.cn
# Generate fake data

import numpy
import logging
import matplotlib.pyplot as plt 
from src.tools.plot import get_ax_data, recreate_axes

# @param data: [List|numpy.ndarray] Data with any size or shape in form of List or numpy.ndarray
# @param mean_shift: [Float] Default 0, the mean of shifted value of forged data
# @param std_shift: [Float] Default 1, the std of shifted value of forged data
# @param dist_shift: [Str] Distribution of shifted value, e.g. "normal", "uniform", "laplace", default normal distribution, 
# @param pos_multiplier: [Float] Default 0, refers to the shift value do not accumulate with positional index
# - If `pos_multiplier > 0`: the shifted value will gradually increase
# - If `pos_multiplier < 0`: the shifted value will gradually decrease
# @param pos_axis: [Int] Which axis Positional multiplier accumulates along, default the last axis -1
# @param dtype: [Str|numpy.dtype] e.g. "int", numpy.dtype("int64"), "float", numpy.dtype("float64")
# @return forged_data: [numpy.ndarray]
def forge_data(data, 
               mean_shift = 0., 
               std_shift = 1.,
               dist_shift = "normal",
               pos_multiplier = 0.,
               pos_axis = -1,
               dtype = None,
               ):
    # Transform data to numpy.ndarray
    if isinstance(data, list):
        data = numpy.array(data, dtype=dtype)
    dtype = str(data.dtype)
    shape = data.shape
    if shape[0] == 0:
        logging.warning("Empty data to be forged!")
        return data
    # Whether the dtype of data is Integer or not
    if "int" in dtype:
        flag_dtype = True	
    elif "float" in dtype:
        flag_dtype = False
    else:
        raise Exception(f"Unknown dtype of keyword argument `data`: {dtype}")
    # Determine the distribution parameters
    assert dist_shift in dir(numpy.random), f"Unknown distribution: {dist_shift}"
    if dist_shift in ["normal", "laplace", "logistic"]:
        dist_kwargs = {"loc": mean_shift, "scale": std_shift, "size": shape}
    elif dist_shift in ["lognormal"]:
        dist_kwargs = {"mean": mean_shift, "sigma": std_shift, "size": shape}
    elif dist_shift in ["uniform"]:
        width = numpy.sqrt(12 * std_shift ** 2)
        low = mean_shift - width / 2
        high = mean_shift + width / 2
        dist_kwargs = {"low": low, "high": high, "size": shape}
    else:
        raise NotImplementedError(dist_shift)
    data_shift = eval(f"numpy.random.{dist_shift}")(**dist_kwargs)
    pos_index_along_axis = numpy.ogrid[tuple(slice(s) for s in shape)]
    multiplier_array = 1.0 + pos_multiplier * pos_index_along_axis[pos_axis] / shape[pos_axis]
    data_shift *= multiplier_array
    forged_data = data + data_shift
    return forged_data
     
# Forge the data of axes to generate another corresponding axes
# @param fig: Matplotlib.figure.Figure
# @param axes: numpy.ndarray of matplotlib subplot axes object
# @param mean_shift: [Float] Default 0, the mean of shifted value of forged data
# @param std_shift: [Float] Default 1, the std of shifted value of forged data
# @param dist_shift: [Str] Distribution of shifted value, e.g. "normal", "uniform", "laplace", default normal distribution, 
# @param pos_multiplier: [Float] Default 0, refers to the shift value do not accumulate with positional index
# - If `pos_multiplier > 0`: the shifted value will gradually increase
# - If `pos_multiplier < 0`: the shifted value will gradually decrease
# @param pos_axis: [Int] Which axis Positional multiplier accumulates along, default the last axis -1
# @param dtype: [Str|numpy.dtype] e.g. "int", numpy.dtype("int64"), "float", numpy.dtype("float64")
# @param save_path: [Str] Figure save path
# @param is_show: [Boolean] Whether to show figure
# @return forged_fig: Matplotlib.figure.Figure
# @return forged_axes: numpy.ndarray of matplotlib subplot axes object
def forge_plot(fig,
               axes,
               mean_shift = 0., 
               std_shift = 1.,
               dist_shift = "normal",
               pos_multiplier = 0.,
               pos_axis = -1,
               dtype = None,
               save_path = None,
               is_show = True,
               ):
    nrows, ncols = axes.shape
    print(nrows, ncols)
    figsize = fig.get_size_inches()	# e.g. array([16., 12.])
    forged_fig, forged_axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    forge_kwargs = {
        "mean_shift": mean_shift, 
        "std_shift": std_shift,
        "dist_shift": dist_shift,
        "pos_multiplier": pos_multiplier,
        "pos_axis": pos_axis,
        "dtype": dtype,
    }
    for row in range(nrows):
        for col in range(ncols):
            if nrows == 1 and ncols == 1:
                source_ax, target_ax = axes, forged_axes
            elif nrows == 1:
                source_ax, target_ax = axes[col], forged_axes[col]
            elif ncols == 1:
                source_ax, target_ax = axes[row], forged_axes[row]
            else:
                source_ax, target_ax = axes[row][col], forged_axes[row][col]
            source_ax_data = get_ax_data(ax=source_ax)
            source_ax_data["ylim"] = (source_ax_data["ylim"][0] + mean_shift, source_ax_data["ylim"][1] + mean_shift)
            for i in range(len(source_ax_data["lines_data"])):
                source_ax_data["lines_data"][i]["ydata"] = forge_data(data = source_ax_data["lines_data"][i]["ydata"], **forge_kwargs)
            for i in range(len(source_ax_data["collections_data"])):
                source_ax_data["collections_data"][i]["array"] = forge_data(source_ax_data["collections_data"][i]["array"], **forge_kwargs)
            if source_ax_data["patches_data"]:
                height_data = [patch_data["height"] for patch_data in source_ax_data["patches_data"]]
                forged_height_data = forge_data(height_data, **forge_kwargs)
                for i in range(len(source_ax_data["patches_data"])):
                    source_ax_data["patches_data"][i]["height"] = forged_height_data[i]
            recreate_axes(target_ax=target_ax, source_ax=None, ax_data=source_ax_data)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()
    if is_show:
        plt.show()
        plt.close()
    return forged_fig, forged_axes
