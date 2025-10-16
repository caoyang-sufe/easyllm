# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@stu.sufe.edu.cn

import os
import json
import time
import logging
from transformers import AutoConfig, AutoTokenizer

from src.tools.datasets import add_dataset_split
from src.unittests import model_home, dataset_home, model_names, dataset_names
from src.pipelines.trainer import base_pipeline, sft_pipeline, ppo_pipeline, dpo_pipeline, grpo_pipeline

# ----------------------------------------------------------------------
# Concrete dataset and model test
def sft_train_math_500(model_id=10, overwritten_model_class="ParallelLlamaForCausalLM", n_cuda=2, adapter_output_dirs=None):
	sft_pipeline_test(
		model_id = model_id,
		train_dataset_id = 5,
		dataset_train_split = "test[:400]",
		test_dataset_ids_and_splits = [(5, "test[400:]")],
		train_data_processor = lambda _data: {"prompt": _data["problem"], "completion": _data["answer"]},
		test_data_processors = [
			lambda _data: {"prompt": _data["problem"], "completion": _data["answer"]},
		],
		overwritten_model_class = overwritten_model_class,
		n_cuda = n_cuda,
		adapter_output_dirs = adapter_output_dirs,
		per_device_train_batch_size = 8,
		per_device_eval_batch_size = 8,
		num_train_epochs = 32,
	)

def sft_train_gsm8k(model_id=10, overwritten_model_class="ParallelLlamaForCausalLM", n_cuda=2, adapter_output_dirs=None):
	sft_pipeline_test(
		model_id = model_id,
		train_dataset_id = 4,
		dataset_train_split = "train",
		test_dataset_ids_and_splits = [(4, "test"), (5, "test")],
		train_data_processor = lambda _data: {"prompt": _data["question"], "completion": _data["answer"]},
		test_data_processors = [
			lambda _data: {"prompt": _data["question"], "completion": _data["answer"]},
			lambda _data: {"prompt": _data["problem"], "completion": _data["answer"]},
		],
		overwritten_model_class = overwritten_model_class,
		n_cuda = n_cuda,
		adapter_output_dirs = adapter_output_dirs,
		per_device_train_batch_size = 8,
		per_device_eval_batch_size = 8,
		num_train_epochs = 32,
	)

def sft_train_leetcodedataset(model_id=10, overwritten_model_class="ParallelLlamaForCausalLM", n_cuda=2, adapter_output_dirs=None):
	sft_pipeline_test(
		model_id = model_id,
		train_dataset_id = 6,
		dataset_train_split = "train",
		test_dataset_ids_and_splits = [(6, "test"), (5, "test")],
		train_data_processor = lambda _data: {"prompt": _data["query"], "completion": _data["response"]},
		test_data_processors = [
			lambda _data: {"prompt": _data["query"], "completion": _data["response"]},
			lambda _data: {"prompt": _data["problem"], "completion": _data["answer"]},
		],
		overwritten_model_class = overwritten_model_class,
		n_cuda = n_cuda,
		adapter_output_dirs = adapter_output_dirs,
		per_device_train_batch_size = 8,
		per_device_eval_batch_size = 8,
		num_train_epochs = 32,
	)

def sft_train_chinese_poems(model_id=10, overwritten_model_class=None, n_cuda=2, adapter_output_dirs=None):
	sft_pipeline_test(
		model_id = model_id,
		train_dataset_id = 7,
		dataset_train_split = "train[:1000]",
		test_dataset_ids_and_splits = [(7, "train[1000:1100]"), (5, "test")],
		train_data_processor = lambda _data: {"prompt": _data["content"][:-10], "completion": _data["content"][-10:]},
		test_data_processors = [
			lambda _data: {"prompt": _data["content"][:-10], "completion": _data["content"][-10:]},
			lambda _data: {"prompt": _data["problem"], "completion": _data["answer"]},
		],
		overwritten_model_class = overwritten_model_class,
		n_cuda = n_cuda,
		adapter_output_dirs = adapter_output_dirs,
		per_device_train_batch_size = 8,
		per_device_eval_batch_size = 8,
		num_train_epochs = 32,
	)

# ----------------------------------------------------------------------
# Base pipeline test
def sft_pipeline_test(
	model_id = 10,
	train_dataset_id = 7,
	dataset_train_split = "train[:1000]",
	test_dataset_ids_and_splits = [(7, "train[1000:1100]"), (5, "test")],
	train_data_processor = lambda _data: {"prompt": _data["content"][:-10], "completion": _data["content"][-10:]},
	test_data_processors = [
		lambda _data: {"prompt": _data["content"][:-10], "completion": _data["content"][-10:]},
		lambda _data: {"prompt": _data["problem"], "completion": _data["answer"]},
	],
	overwritten_model_class = "ParallelLlamaForCausalLM",
	n_cuda = 2,
	adapter_output_dirs = None,
	per_device_train_batch_size = 8,
	per_device_eval_batch_size = 8,
	num_train_epochs = 32,
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
			# [num_hidden_layers - 1],	# Tail 0
			[0],	# Head 0
			[num_hidden_layers - 1, num_hidden_layers - 2],	# Tail 1
			[0, 1],	# Head 1
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
		}
		kwargs = {**{"adapter_output_dirs": adapter_output_dirs}, **config_kwargs, **trainer_kwargs}
		os.makedirs(config_kwargs["output_dir"], exist_ok=True)
		with open(os.path.join(config_kwargs["output_dir"], "kwargs.json"), 'w', encoding="utf8") as f:
			json.dump(kwargs, f, ensure_ascii=False)
		sft_pipeline(
			train_data_processor,
			test_data_processors,
			config_kwargs,
			trainer_kwargs,
			overwritten_model_class = overwritten_model_class,
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

