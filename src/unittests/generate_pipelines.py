# -*- coding: utf8 -*-
# @author: caoyang
# @email: caoyang@stu.sufe.edu.cn

import os
import torch
import pandas
import logging

from src.unittests import model_home, dataset_home, model_names, dataset_names
from src.pipelines.generate import decode_pipeline, generate_pipeline

def decode_pipeline_test():
	logging.info("Decode unittest ...")
	model_id = 0
	model_name_or_path = os.path.join(model_home, model_names[model_id])
	logging.info(f"  - Model: {model_name_or_path}")
	prompt = """英文单词strawberry中有几个字母s？"""
	max_length = 32
	use_kv_cache = True
	
	forward_hook_module_names = \
		[f"model.layers[{i}].self_attn.q_proj" for i in range(24)] + \
		[f"model.layers[{i}].self_attn.k_proj" for i in range(24)] + \
		[f"model.layers[{i}].self_attn.v_proj" for i in range(24)]
	returned_dict = decode_pipeline(
		model_name_or_path,
		prompt,
		max_length,
		device = None,
		use_kv_cache = use_kv_cache,
		forward_hook_module_names = forward_hook_module_names,
		backward_hook_module_names = None,
	)
	df_display = returned_dict["df_display"]
	forward_hook_data = returned_dict["forward_hook_data"]
	backward_hook_data = returned_dict["backward_hook_data"]
	# Save returned data
	save_path = f"./decode+{model_names[model_id].split('/')[-1]}+{use_kv_cache}.csv"
	logging.info(f"Export table to {save_path}")
	df_display.to_csv(save_path, sep='\t', header=True, index=False)
	if forward_hook_data is not None:
		save_path = f"./fhook+{model_names[model_id].split('/')[-1]}+{use_kv_cache}.pt"
		logging.info(f"Export forward hook data to {save_path}")
		torch.save(forward_hook_data, save_path)
	if backward_hook_data is not None:
		save_path = f"./bhook+{model_names[model_id].split('/')[-1]}+{use_kv_cache}.pt"
		logging.info(f"Export backward hook data to {save_path}")
		torch.save(backward_hook_data, save_path)
	logging.info("  - OK!")

def generate_pipeline_test():
	logging.info("Generate unittest ...")
	model_id = 0
	model_name_or_path = os.path.join(model_home, model_names[model_id])
	logging.info(f"  - Model: {model_name_or_path}")
	prompt = """英文单词strawberry中有几个字母r？"""
	max_length = 40

	df_display = generate_pipeline(
		model_name_or_path,
		prompt,
		max_length,
		device = None,
		generate_kwargs = None,
	)
	save_path = f"./generate+{model_names[model_id].split('/')[-1]}.csv"
	logging.info(f"Export to {save_path}")
	df_display.to_csv(save_path, sep='\t', header=True, index=False)
	logging.info("  - OK!")
