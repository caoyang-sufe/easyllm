# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@stu.sufe.edu.cn

import os
import json
import time
import logging
from transformers import AutoConfig, AutoTokenizer

from src.tools.easy import save_args
from src.tools.datasets import add_dataset_split
from src.tools.metric import generate_compute_metrics_function
from src.unittests import (
	model_home, dataset_home, model_names, dataset_names, evaluate_home, 
	dataset_processors_map, dataset_train_test_splits_map, model_parallel_classes_map
)
from src.pipelines.trainer import base_pipeline, sft_pipeline, ppo_pipeline, dpo_pipeline, grpo_pipeline

# Base pipeline test
def sft_pipeline_test(
	model_id = 10,
	train_dataset_id = 7,
	eval_dataset_ids = [7, 5],
	adapter_output_dirs = None,
	per_device_train_batch_size = 8,
	per_device_eval_batch_size = 8,
	num_train_epochs = 32,
	n_cuda = 2,
	use_overwritten_model_class = True,
):
	model_name_or_path = os.path.join(model_home, model_names[model_id])
	model_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
	num_hidden_layers = model_config.num_hidden_layers
	if adapter_output_dirs is None:
		# 1-stage-sft
		target_layer_ids_list = [
			list(range(num_hidden_layers)),	# Full
			[0, 1, 2, num_hidden_layers - 3, num_hidden_layers - 2, num_hidden_layers - 1],	# Head and tails only
			list(range(3, num_hidden_layers - 3)),	# Body only
			list(range(num_hidden_layers // 2)), # 1st Half
			list(range(num_hidden_layers // 2, num_hidden_layers)), # 2nd Half
		]
	else:
		# 2-stage-sft
		target_layer_ids_list = [
			[num_hidden_layers - 1],	# Tail 1
			[0],	# Head 1
			[num_hidden_layers - 1, num_hidden_layers - 2],	# Tail 2
			[0, 1],	# Head 2
			# [num_hidden_layers - 1, num_hidden_layers - 2, num_hidden_layers - 3],	# Tail 3
			# [0, 1, 2],	# Head 3
		]
	dataset_train_split = dataset_train_test_splits_map[train_dataset_id]["train"]
	test_dataset_ids_and_splits = [
		(eval_dataset_id, dataset_train_test_splits_map[eval_dataset_id]["test"])
		for eval_dataset_id in eval_dataset_ids
	]
	train_data_processor = dataset_processors_map[train_dataset_id]["train"]
	test_data_processors = [
		dataset_processors_map[eval_dataset_id]["test"]
		for eval_dataset_id in eval_dataset_ids
	]
	for target_layer_ids in target_layer_ids_list:
		time_string = time.strftime("%Y%m%d%H%M%S")
		logging.info(f"Experiment on `target_layer_ids`: {target_layer_ids}")
		train_dataset_path = os.path.join(dataset_home, dataset_names[train_dataset_id])
		test_dataset_paths = [os.path.join(dataset_home, dataset_names[test_dataset_id]) for (test_dataset_id, test_dataset_split_name) in test_dataset_ids_and_splits]
		dataset_test_splits = [test_dataset_split_name for (test_dataset_id, test_dataset_split_name) in test_dataset_ids_and_splits]
		dataset_name = {"train": train_dataset_path, "test": test_dataset_paths}
		logging.info(f"  - Model: {model_name_or_path}")
		logging.info(f"  - Dataset: {dataset_name}")
		config_kwargs = {
			"output_dir": f"./temp/sft+{model_name_or_path.split('/')[-1]}+{train_dataset_path.split('/')[-1]}+{time_string}",
			"model_name_or_path": model_name_or_path,
			"dataset_name": dataset_name,
			"trust_remote_code": True,
			"dataset_train_split": dataset_train_split,
			"dataset_test_split": dataset_test_splits,
			"use_peft": True,
			"report_to": "none",
			# LoRA
			"lora_target_modules": [f"model.layers.{i}.self_attn.q_proj" for i in target_layer_ids] + \
				[f"model.layers.{i}.self_attn.k_proj" for i in target_layer_ids] + \
				[f"model.layers.{i}.self_attn.v_proj" for i in target_layer_ids],
			"lora_r": 16,
			"lora_alpha": 16,
			"lora_dropout": .05,
			"lora_task_type": "CAUSAL_LM",
			# Logging
			"logging_strategy": "steps",
			"logging_steps": 1,
			"eval_strategy": "epoch",
			"save_strategy": "epoch",
			# Train
			"per_device_train_batch_size": per_device_train_batch_size,
			"per_device_eval_batch_size": per_device_eval_batch_size,
			"num_train_epochs": num_train_epochs,
		}
		trainer_kwargs = {
			# "compute_metrics": generate_compute_metrics_function(metrics = ["bleu", "rouge"],
																 # strategy = "evaluate",
																 # evaluate_home = evaluate_home,
																 # tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True),
																 # ),
			# "compute_metrics": generate_compute_metrics_function(metrics = [("calc_bleu", {"min_grams": 1, "max_grams": 3}, "bleu_3"),
																			# ("calc_rouge_n", {'n': 3, "beta": 1}, "rouge_3"),
																			# ("calc_rouge_w", {"weight_function": lambda _x: _x, "weight_function_reverse": lambda _x: _x, "beta": 1}, "rouge_l"),
																			# ("calc_rouge_w", {"weight_function": lambda _x: _x ** 2, "weight_function_reverse": lambda _x: _x ** 0.5, "beta": 1}, "rouge_w"),
																			# ],
																 # strategy = "diy",
																 # evaluate_home = evaluate_home,
																 # tokenizer = None,
																 # ),
		}
		kwargs = {**{"adapter_output_dirs": adapter_output_dirs}, **config_kwargs, **trainer_kwargs}
		os.makedirs(config_kwargs["output_dir"], exist_ok=True)
		save_args(kwargs, save_path = os.path.join(config_kwargs["output_dir"], "kwargs.json"))
		sft_pipeline(
			train_data_processor,
			test_data_processors,
			config_kwargs,
			trainer_kwargs,
			overwritten_model_class = model_parallel_classes_map[model_id] if use_overwritten_model_class else None,
			overwritten_model_class_init_kwargs = {"n_cuda": n_cuda, "device_map": "cpu"},
			adapter_output_dirs = adapter_output_dirs,
			parse_arguments = False,
		)

def ppo_pipeline_test():
	logging.info("PPO unittest ...")
	# model_name_or_path = os.path.join(model_home, model_names[1])
	# """
	# EleutherAI/pythia-1b-deduped
	# GPTNeoXForCausalLM(
	  # (gpt_neox): GPTNeoXModel(
		# (embed_in): Embedding(50304, 2048)
		# (emb_dropout): Dropout(p=0.0, inplace=False)
		# (layers): ModuleList(
		  # (0-15): 16 x GPTNeoXLayer(
			# (input_layernorm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
			# (post_attention_layernorm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
			# (post_attention_dropout): Dropout(p=0.0, inplace=False)
			# (post_mlp_dropout): Dropout(p=0.0, inplace=False)
			# (attention): GPTNeoXAttention(
			  # (query_key_value): Linear(in_features=2048, out_features=6144, bias=True)
			  # (dense): Linear(in_features=2048, out_features=2048, bias=True)
			# )
			# (mlp): GPTNeoXMLP(
			  # (dense_h_to_4h): Linear(in_features=2048, out_features=8192, bias=True)
			  # (dense_4h_to_h): Linear(in_features=8192, out_features=2048, bias=True)
			  # (act): GELUActivation()
			# )
		  # )
		# )
		# (final_layer_norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
		# (rotary_emb): GPTNeoXRotaryEmbedding()
	  # )
	  # (embed_out): Linear(in_features=2048, out_features=50304, bias=False)
	# )
	# """
	# dataset_name = os.path.join(dataset_home, dataset_names[0])
	# reward_model_path = os.path.join(model_home, model_names[2])
	# logging.info(f"  - Model: {model_name_or_path}")
	# logging.info(f"  - Dataset: {dataset_name}")
	# logging.info(f"  - Reward: {reward_model_path}")
	# data_processor = None
	# config_kwargs = {
		# "output_dir": f"./temp/ppo+{model_name_or_path.split('/')[-1]}+{dataset_name.split('/')[-1]}",
		# "model_name_or_path": model_name_or_path,
		# "dataset_name": dataset_name,
		# "reward_model_path": reward_model_path,
		# "trust_remote_code": True,
		# "dataset_train_split": "train[:500]",
		# "dataset_test_split": "validation[:100]",
		# # "lr_scheduler_type": "cosine",	# Default "linear"
		# "use_peft": True,
		# "report_to": "none",
		# "lora_target_modules": ["query_key_value"],
	# }
	# trainer_kwargs = {
	# }
	# ppo_pipeline(data_processor, config_kwargs, trainer_kwargs)

	model_name_or_path = os.path.join(model_home, model_names[0])
	dataset_name = os.path.join(dataset_home, dataset_names[0])
	reward_model_path = os.path.join(model_home, model_names[3])
	logging.info(f"  - Model: {model_name_or_path}")
	logging.info(f"  - Dataset: {dataset_name}")
	logging.info(f"  - Reward: {reward_model_path}")
	data_processor = None
	config_kwargs = {
		"output_dir": f"./temp/ppo+{model_name_or_path.split('/')[-1]}+{dataset_name.split('/')[-1]}",
		"model_name_or_path": model_name_or_path,
		"dataset_name": dataset_name,
		"reward_model_path": reward_model_path,
		"trust_remote_code": True,
		"dataset_train_split": "train[:500]",
		"dataset_test_split": "validation[:100]",
		# "lr_scheduler_type": "cosine",	# Default "linear"
		"use_peft": True,
		"report_to": "none",
		"lora_target_modules": ["q_proj", "k_proj", "v_proj"],
	}
	trainer_kwargs = {
	}
	ppo_pipeline(data_processor, config_kwargs, trainer_kwargs)

def dpo_pipeline_test():
	logging.info("DPO unittest ...")
	model_name_or_path = os.path.join(model_home, model_names[0])
	dataset_name = os.path.join(dataset_home, dataset_names[2])
	logging.info(f"  - Model: {model_name_or_path}")
	logging.info(f"  - Dataset: {dataset_name}")
	data_processor = None
	config_kwargs = {
		"output_dir": f"./temp/dpo+{model_name_or_path.split('/')[-1]}+{dataset_name.split('/')[-1]}",
		"model_name_or_path": model_name_or_path,
		"dataset_name": dataset_name,
		"trust_remote_code": True,
		"dataset_train_split": "descriptiveness[:500]",
		"dataset_test_split": "descriptiveness[500:600]",
		# "lr_scheduler_type": "cosine",	# Default "linear"
		"use_peft": True,
		"report_to": "none",
		"lora_target_modules": ["q_proj", "k_proj", "v_proj"]
	}
	trainer_kwargs = {
	}
	dpo_pipeline(data_processor, config_kwargs, trainer_kwargs)

def grpo_pipeline_test():
	logging.info("GRPO unittest ...")
	model_name_or_path = os.path.join(model_home, model_names[0])
	dataset_name = os.path.join(dataset_home, dataset_names[0])
	logging.info(f"  - Model: {model_name_or_path}")
	logging.info(f"  - Dataset: {dataset_name}")
	data_processor = None
	def reward_funcs(completions, **kwargs):
		return [float(len(set(completion))) for completion in completions]
	config_kwargs = {
		"output_dir": f"./temp/grpo+{model_name_or_path.split('/')[-1]}+{dataset_name.split('/')[-1]}",
		"model_name_or_path": model_name_or_path,
		"dataset_name": dataset_name,
		"trust_remote_code": True,
		"dataset_train_split": "train[:500]",
		"dataset_test_split": "validation[:100]",
		# "lr_scheduler_type": "cosine",	# Default "linear"
		"use_peft": True,
		"report_to": "none",
		"lora_target_modules": ["q_proj", "k_proj", "v_proj"]
	}
	trainer_kwargs = {
		"reward_funcs": reward_funcs,
	}
	grpo_pipeline(data_processor, config_kwargs, trainer_kwargs)

