# -*- coding: utf8 -*-
# @author: caoyang
# @email: caoyang@stu.sufe.edu.cn

import os
import re
import string
import torch
from torch import nn
from torch.nn import functional as F
if not "CHDIR_FLAG" in dir():
    os.chdir("../")
    CHDIR_FLAG = True
    
import numpy as np
import pandas as pd
from datasets import load_dataset

from src.tools.torch import register_forward_hook_decorator, register_backward_hook_decorator
from src.tools.plot import plot_tensor_histogram, plot_tensor_heatmap
from transformers import AutoTokenizer, AutoModelForCausalLM

import matplotlib
from matplotlib import pyplot as plt
%matplotlib inline

from src.pipelines.analysis import horizontal_comparison_of_forward_hook, vertical_comparison_of_forward_hook

def horizontal_comparison_of_forward_hook_test():
	forward_hook_module_names = \
		[f"model.layers[{i}].self_attn.q_proj" for i in range(24)] + \
		[f"model.layers[{i}].self_attn.k_proj" for i in range(24)] + \
		[f"model.layers[{i}].self_attn.v_proj" for i in range(24)]
	hook_data_path_1_1 = r"./results/strawberry-1/fhook+Qwen2.5-0.5B-Instruct+False.pt"
	hook_data_path_1_2 = r"./results/strawberry-1/fhook+Qwen2.5-0.5B-Instruct+True.pt"
	hook_data_path_2_1 = r"./results/strawberry-2/fhook+Qwen2.5-0.5B-Instruct+False.pt"
	hook_data_path_2_2 = r"./results/strawberry-2/fhook+Qwen2.5-0.5B-Instruct+True.pt"
	table_path_1_1 = r"./results/strawberry-1/decode+Qwen2.5-0.5B-Instruct+False.csv"
	table_path_1_2 = r"./results/strawberry-1/decode+Qwen2.5-0.5B-Instruct+True.csv"
	table_path_2_1 = r"./results/strawberry-2/decode+Qwen2.5-0.5B-Instruct+False.csv"
	table_path_2_2 = r"./results/strawberry-2/decode+Qwen2.5-0.5B-Instruct+True.csv"
	comparison_summary_dict = horizontal_comparison_of_forward_hook(
		hook_datas = None,
		hook_data_paths = [hook_data_path_1_2, hook_data_path_2_2],
		hook_module_names = forward_hook_module_names[:],
		hook_module_name_suffixes = ["q_proj", "k_proj", "v_proj"],
		comparison_index = ["mean_diff", "max_diff", "corr"],
		max_length = 4,
	)
	comparison_summary_dict = horizontal_comparison_of_forward_hook(
		hook_datas = None,
		hook_data_paths = [hook_data_path_1_2, hook_data_path_2_2],
		hook_module_names = forward_hook_module_names[:],
		hook_module_name_suffixes = ["q_proj"],
		comparison_index = ["mean_diff", "max_diff", "corr"],
		max_length = 4,
	)
	comparison_summary_dict = horizontal_comparison_of_forward_hook(
		hook_datas = None,
		hook_data_paths = [hook_data_path_1_2, hook_data_path_2_2],
		hook_module_names = forward_hook_module_names[:],
		hook_module_name_suffixes = ["q_proj", "k_proj", "v_proj"],
		comparison_index = ["corr"],
		max_length = 4,
	)
	comparison_summary_dict = horizontal_comparison_of_forward_hook(
		hook_datas = None,
		hook_data_paths = [hook_data_path_1_2, hook_data_path_2_2],
		hook_module_names = forward_hook_module_names[:],
		hook_module_name_suffixes = ["q_proj"],
		comparison_index = ["corr"],
		max_length = 4,
	)


def vertical_comparison_of_forward_hook_test():
	vertical_comparison_of_forward_hook(
		hook_data = None,
		hook_data_path = hook_data_paths[5],
		hook_module_names = [f"model.layers[{i}]" for i in range(24)],
		comparison_index = ["mean_diff", "max_diff", "corr"],
		max_length = 16,
		figure_size = 5,
		watched_module_names = [f"model.layers[{i}]" for i in [0, 1, 4, 5]],
	)
	vertical_comparison_of_forward_hook(
		hook_data = None,
		hook_data_path = hook_data_paths[5],
		hook_module_names = [f"model.layers[{i}]" for i in range(24)],
		comparison_index = ["mean_diff"],
		max_length = 16,
		figure_size = 5,
	)
