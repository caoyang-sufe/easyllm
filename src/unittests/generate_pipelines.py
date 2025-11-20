# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@stu.sufe.edu.cn

import os
import gc
import torch
import string
import pandas
import logging
from transformers import AutoConfig, AutoTokenizer, AutoModel, AutoModelForCausalLM
from src.unittests import MODEL_HOME, DATASET_HOME, MODEL_NAMES, DATASET_NAMES, LONG_PROMPT
from src.pipelines.generate import one_time_forward_pipeline, decode_pipeline, generate_pipeline
from src.modules import (
	ParallelQwen2Model, SkipLayerQwen2ForCausalLM,
	ParallelQwen2ForCausalLM, SkipLayerQwen2ForCausalLM,
	ParallelQwen3Model, SkipLayerQwen3ForCausalLM,
	ParallelQwen3ForCausalLM, SkipLayerQwen3ForCausalLM,
	ParallelLlamaModel, SkipLayerLlamaForCausalLM,
	ParallelLlamaForCausalLM, SkipLayerLlamaForCausalLM,
	SkipLayerDeepseekModel, SkipLayerDeepseekForCausalLM,
	ParallelDeepseekModel, ParallelDeepseekForCausalLM,
	SkipLayerDeepseekV2Model, SkipLayerDeepseekV2ForCausalLM,
	ParallelDeepseekV2Model, ParallelDeepseekV2ForCausalLM,
	SkipLayerDeepseekV3Model, SkipLayerDeepseekV3ForCausalLM,
	ParallelDeepseekV3Model, ParallelDeepseekV3ForCausalLM,
)

# Do one forward for long prompts
def one_time_forward_pipeline_test(model_id=-1, prompts=None, device=None, overwritten_model_class=None, n_cuda=2):
	logging.info("One time forward unittest")
	model_name_or_path = os.path.join(MODEL_HOME, MODEL_NAMES[model_id])
	model_name = MODEL_NAMES[model_id].split('/')[-1].split('\\')[-1]
	model_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
	num_hidden_layers = model_config.num_hidden_layers
	forward_hook_module_names = [f"layers[{i}]" for i in range(num_hidden_layers)]
	backward_hook_module_names = None
	if prompts is None:
		prompts = LONG_PROMPT[:]
	logging.info(f"Load model from: {model_name_or_path}")
	if overwritten_model_class is None:
		logging.info("  - Using AutoModel ...")
		model = AutoModel.from_pretrained(model_name_or_path, device_map="auto")
	else:
		logging.info(f"  - Using {overwritten_model_class} ...")
		model = eval(overwritten_model_class).from_pretrained(
			model_name_or_path,
			n_cuda = n_cuda,
			device_map = "cpu",
		)
		# model.module_to_device()	# Currently do module_to_device at the first forward propagation
	tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
	for name, parameter in model.named_parameters():
		logging.info(f"{name}: {parameter.device}")
	for i in range(len(prompts)):
		logging.info(f"Forward prompt {i}")
		try:
			hook_data = one_time_forward_pipeline(
				model = model,
				tokenizer = tokenizer,
				prompt = prompts[i],
				device = "cuda:0",
				forward_hook_module_names = forward_hook_module_names,
				backward_hook_module_names = backward_hook_module_names,
			)
			save_path = f"./temp/1f+fhook+{model_name}+{i}.pt"
			logging.info(f"Export forward hook data to {save_path}")
			torch.save(hook_data, save_path)
			del hook_data
			gc.collect()
		except Exception as exception:
			logging.warning(f"Error: {exception}")
			gc.collect()
			continue

# Test `src.pipelines.generate.decode_pipeline`
def decode_pipeline_test(model_id=-1, device=None, overwritten_model_class=None, n_cuda=2):
	logging.info("Decode unittest ...")
	model_name_or_path = os.path.join(MODEL_HOME, MODEL_NAMES[model_id])
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

	if "meta-llama" in model_name_or_path:
		logging.info("Use English prompts ...")
		prompts = [
			f"""Once upon a time""",
			f"""Solve the equation: x^2 - 3x + 2 = 0""",
			f"""Write a bubble sort algorithm using Python""",
			f"""I am 20 years old now, and my sister is half my age. How old will my sister be when I am 40?""",
			f"""Prime factorization: 126.""",
			f"""Please use markdown syntax to create a 3-row, 4-column table. The headers should be "Name", "Age", and "Gender". For the remaining 3 rows, please randomly generate names, ages, and genders for 3 fictional characters.""",
			f"""Write a 12-line English poem celebrating the 80th anniversary of a nation's founding. Use iambic pentameter with an ABAB rhyme scheme for the first two stanzas, and incorporate themes of unity, progress, and hope for the future. Include at least three poetic devices such as metaphor, alliteration, or personification. The tone should be uplifting and ceremonial.
Ode to Eighty Years"""
		]

	else:
		logging.info("Use Chinese prompts ...")
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
	# forward_hook_module_names = None
	for i in range(len(prompts)):
		returned_dict = decode_pipeline(
			model_name_or_path,
			prompts[i],
			max_length,
			device = device,
			use_kv_cache = use_kv_cache,
			forward_hook_module_names = forward_hook_module_names,
			backward_hook_module_names = None,
			overwritten_model_class = overwritten_model_class,
			n_cuda = n_cuda,
		)
		df_display = returned_dict["df_display"]
		forward_hook_data = returned_dict["forward_hook_data"]
		backward_hook_data = returned_dict["backward_hook_data"]
		# Save returned data
		model_name = MODEL_NAMES[model_id].split('/')[-1].split('\\')[-1]
		save_path = f"./temp/decode+{model_name}+{use_kv_cache}-{i}.csv"
		logging.info(f"Export table to {save_path}")
		df_display.to_csv(save_path, sep='\t', header=True, index=False)
		if forward_hook_data is not None:
			save_path = f"./temp/fhook+{model_name}+{use_kv_cache}-{i}.pt"
			logging.info(f"Export forward hook data to {save_path}")
			torch.save(forward_hook_data, save_path)
		if backward_hook_data is not None:
			save_path = f"./temp/bhook+{model_name}+{use_kv_cache}-{i}.pt"
			logging.info(f"Export backward hook data to {save_path}")
			torch.save(backward_hook_data, save_path)
		logging.info("  - OK!")

# Test `src.pipelines.generate.generate_pipeline`
def generate_pipeline_test(model_id=-1, device=None, overwritten_model_class=None, n_cuda=2):
	logging.info("Generate unittest ...")
	model_name_or_path = os.path.join(MODEL_HOME, MODEL_NAMES[model_id])
	logging.info(f"  - Model: {model_name_or_path}")
	prompt = """英文单词strawberry中有几个字母r？"""
	max_length = 40
	df_display = generate_pipeline(
		model_name_or_path,
		prompt,
		max_length,
		device = device,
		generate_kwargs = None,
		overwritten_model_class = overwritten_model_class,
		n_cuda = n_cuda,
	)
	save_path = f"./generate+{MODEL_NAMES[model_id].split('/')[-1]}.csv"
	logging.info(f"Export to {save_path}")
	df_display.to_csv(save_path, sep='\t', header=True, index=False)
	logging.info("  - OK!")
