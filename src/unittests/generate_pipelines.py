# -*- coding: utf8 -*-
# @author: caoyang
# @email: caoyang@stu.sufe.edu.cn

import os
import torch
import string
import pandas
import logging

from transformers import AutoConfig

from src.unittests import model_home, dataset_home, model_names, dataset_names
from src.pipelines.generate import decode_pipeline, generate_pipeline

def decode_pipeline_test():
	logging.info("Decode unittest ...")
	model_id = 8
	model_name_or_path = os.path.join(model_home, model_names[model_id])
	model_config = AutoConfig.from_pretrained(model_name_or_path)
	num_hidden_layers = model_config.num_hidden_layers
	
	logging.info(f"  - Model: {model_name_or_path}")
	# prompts = \
		# [f"""英文单词strawberry中有几个字母{i}？""" for i in string.ascii_letters] + \
		# [f"""({i})英文单词strawberry中有几个字母{i}？""" for i in range(1, 10)] + \
		# [f"""({i})英文单词strawberry中有几个字母{i}？""" for i in string.ascii_letters] + \
		# [
			# """（ii）英文单词strawberry中有几个字母r？""",
			# """（iii）英文单词strawberry中有几个字母r？""",
			# """（iv）英文单词strawberry中有几个字母r？""",
			# """（vi）英文单词strawberry中有几个字母r？""",
			# """（vii）英文单词strawberry中有几个字母r？""",
			# """（viii）英文单词strawberry中有几个字母r？""",
			# """（ix）英文单词strawberry中有几个字母r？""",
		# ]
	# prompts = [f"""英文单词strawberry中有几个字母{i}？""" for i in string.ascii_letters]
	# prompts = [f"""很久很久以前，"""]
	# prompts = [
		# f"""英文单词strawberry中有几个字母r？<think>""",
		# f"""很久很久以前，""",
		# f"""素因子分解：512<think>""",
		# f"""请使用markdown语法编写一个3行4列的表格，表头为“姓名”、“年龄”、“性别”，剩余3行请随机构造3个人物的姓名、年龄以及性别填写。<think>""",
	# ]
	
	prompts = [
		f"""很久很久以前""",
		f"""解方程：x^2 - 3x + 2 = 0""",
		f"""使用python写一段冒泡排序算法""",
		f"""我今年20岁，妹妹的年龄是我的一半。则我40岁时，我的妹妹多少岁？""",
		f"""素因子分解：126。""",
		f"""请使用markdown语法编写一个3行4列的表格，表头为“姓名”、“年龄”、“性别”，剩余3行请随机构造3个人物的姓名、年龄以及性别填写。""",
		f"""请写一首七言律诗作为中华人民共和国成立八十周年的祝词，注意用词的平仄押韵：
《八十周年庆》"""
	]
	
	max_length = 64
	use_kv_cache = True
	forward_hook_module_names = \
		[f"model.embed_tokens", "lm_head"] + \
		[f"model.layers[{i}].self_attn.q_proj" for i in range(num_hidden_layers)] + \
		[f"model.layers[{i}].self_attn.k_proj" for i in range(num_hidden_layers)] + \
		[f"model.layers[{i}].self_attn.v_proj" for i in range(num_hidden_layers)] + \
		[f"model.layers[{i}].self_attn.o_proj" for i in range(num_hidden_layers)]
	forward_hook_module_names = [f"model.layers[{i}]" for i in range(num_hidden_layers)]
	for i in range(len(prompts)):
		returned_dict = decode_pipeline(
			model_name_or_path,
			prompts[i],
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
		save_path = f"./temp/decode+{model_names[model_id].split('/')[-1]}+{use_kv_cache}-{i}.csv"
		logging.info(f"Export table to {save_path}")
		df_display.to_csv(save_path, sep='\t', header=True, index=False)
		if forward_hook_data is not None:
			save_path = f"./temp/fhook+{model_names[model_id].split('/')[-1]}+{use_kv_cache}-{i}.pt"
			logging.info(f"Export forward hook data to {save_path}")
			torch.save(forward_hook_data, save_path)
		if backward_hook_data is not None:
			save_path = f"./temp/bhook+{model_names[model_id].split('/')[-1]}+{use_kv_cache}-{i}.pt"
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
