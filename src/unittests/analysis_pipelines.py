# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@stu.sufe.edu.cn

import os
import re
import torch
import string
import logging
from torch import nn
from torch.nn import functional as F
if not "CHDIR_FLAG" in dir():
    os.chdir("../")
    CHDIR_FLAG = True
    
import numpy as np
import pandas as pd
from datasets import load_dataset

from src.tools.transformers import get_generation_eos_token_ids
from src.tools.torch import register_forward_hook_decorator, register_backward_hook_decorator
from src.tools.plot import plot_tensor_histogram, plot_tensor_heatmap
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

from src.unittests import model_home, dataset_home, model_names, dataset_names
from src.pipelines.analysis import horizontal_comparison_of_forward_hook, vertical_comparison_of_forward_hook, easy_skip_layer_generation, skip_layer_generation
from src.pipelines.generate import display_pipeline

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
	
def skip_layer_generation_test():
	logging.info("skip layer unittest ...")
	model_id = 8
	model_name_or_path = os.path.join(model_home, model_names[model_id])
	model_config = AutoConfig.from_pretrained(model_name_or_path)
	num_hidden_layers = model_config.num_hidden_layers	

	prompts = [
		f"""英文单词strawberry中有几个字母r？<think>""",
		f"""很久很久以前，""",
		f"""素因子分解：512<think>""",
		f"""请使用markdown语法编写一个3行4列的表格，表头为“姓名”、“年龄”、“性别”，剩余3行请随机构造3个人物的姓名、年龄以及性别填写。<think>""",
	]
	max_length = 64
	use_kv_cache = True
	tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
	model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
	eos_token_ids = get_generation_eos_token_ids(model)
	forward_hook_module_names = [f"model.layers[{j}]" for j in range(num_hidden_layers - 1)]
	for i in range(len(prompts)):
		for skip_layer_id in range(num_hidden_layers):	
			skip_layer_ids = [skip_layer_id]
			results = easy_skip_layer_generation(
				model = model,
				tokenizer = tokenizer,
				prompt = prompts[i],
				max_length = max_length,
				skip_layer_ids = skip_layer_ids,
				forward_hook_module_names = forward_hook_module_names,
			)
			text, token_probs, logits = results["text"], results["token_probs"], results["logits"]
			forward_hook_data, backward_hook_data = results["forward_hook_data"], results["backward_hook_data"]
			logging.info(f"Generated text: {text}")
			df_display = display_pipeline(tokenizer, text, token_probs, logits, eos_token_id=eos_token_ids[0])
			# Save returned data
			save_path = f"./temp/decode+{model_names[model_id].split('/')[-1]}+{use_kv_cache}-{i}-skip{str(skip_layer_id).zfill(2)}.csv"
			logging.info(f"Export table to {save_path}")
			df_display.to_csv(save_path, sep='\t', header=True, index=False)
			if forward_hook_data is not None:
				save_path = f"./temp/fhook+{model_names[model_id].split('/')[-1]}+{use_kv_cache}-{i}-skip{str(skip_layer_id).zfill(2)}.pt"
				logging.info(f"Export forward hook data to {save_path}")
				torch.save(forward_hook_data, save_path)
			if backward_hook_data is not None:
				save_path = f"./temp/bhook+{model_names[model_id].split('/')[-1]}+{use_kv_cache}-{i}-skip{str(skip_layer_id).zfill(2)}.pt"
				logging.info(f"Export backward hook data to {save_path}")
				torch.save(backward_hook_data, save_path)
		logging.info("  - OK!")
