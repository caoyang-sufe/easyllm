# -*- coding: utf8 -*-
# @author: caoyang
# @email: caoyang@stu.sufe.edu.cn

import os
import logging
from src.pipelines.trainer import base_pipeline, sft_pipeline, ppo_pipeline, dpo_pipeline, grpo_pipeline

model_home = "/nfsshare/home/caoyang/resource/model"
dataset_home = "/nfsshare/home/caoyang/resource/dataset"
model_names = [
	"Qwen/Qwen2.5-0.5B-Instruct",
	"EleutherAI/pythia-1b-deduped",
	"EleutherAI/pythia-160m",
]

dataset_names = [
	"trl-lib/tldr",	# train["prompt", "completion"] + validation["prompt", "completion"] + test["prompt", "completion"]
	"trl-lib/ultrafeedback_binarized",	# train["chosen", "rejected", "score_chosen", "score_rejected"] + test["chosen", "rejected", "score_chosen", "score_rejected"]
	"trl-internal-testing/descriptiveness-sentiment-trl-style", # sentiment["prompt", "chosen", "rejected"] + descriptiveness["prompt", "chosen", "rejected"]
	"YeungNLP/firefly-train-1.1M", # train["input", "target"]
]

def sft_pipeline_test():
	logging.info("SFT unittest ...")
	model_name_or_path = os.path.join(model_home, model_names[0])
	dataset_name = os.path.join(dataset_home, dataset_names[0])
	data_processor = None
	config_kwargs = {
		"output_dir": f"./temp/sft+{model_name_or_path.split('/')[-1]}+{dataset_name.split('/')[-1]}",
		"model_name_or_path": model_name_or_path,
		"dataset_name": dataset_name,
		"trust_remote_code": True,
		"dataset_train_split": "train[:500]",
		"dataset_test_split": "validation[500:600]",
		"use_peft": True,
		"report_to": "none",
		"lora_target_modules": ["q_proj", "k_proj", "v_proj"]
	}
	trainer_kwargs = {
	}
	sft_pipeline(data_processor, config_kwargs, trainer_kwargs)

def ppo_pipeline_test():
	logging.info("PPO unittest ...")
	model_name_or_path = os.path.join(model_home, model_names[1])
	"""
	EleutherAI/pythia-1b-deduped
	GPTNeoXForCausalLM(
	  (gpt_neox): GPTNeoXModel(
		(embed_in): Embedding(50304, 2048)
		(emb_dropout): Dropout(p=0.0, inplace=False)
		(layers): ModuleList(
		  (0-15): 16 x GPTNeoXLayer(
			(input_layernorm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
			(post_attention_layernorm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
			(post_attention_dropout): Dropout(p=0.0, inplace=False)
			(post_mlp_dropout): Dropout(p=0.0, inplace=False)
			(attention): GPTNeoXAttention(
			  (query_key_value): Linear(in_features=2048, out_features=6144, bias=True)
			  (dense): Linear(in_features=2048, out_features=2048, bias=True)
			)
			(mlp): GPTNeoXMLP(
			  (dense_h_to_4h): Linear(in_features=2048, out_features=8192, bias=True)
			  (dense_4h_to_h): Linear(in_features=8192, out_features=2048, bias=True)
			  (act): GELUActivation()
			)
		  )
		)
		(final_layer_norm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
		(rotary_emb): GPTNeoXRotaryEmbedding()
	  )
	  (embed_out): Linear(in_features=2048, out_features=50304, bias=False)
	)
	"""
	dataset_name = os.path.join(dataset_home, dataset_names[0])
	reward_model_path = os.path.join(model_home, model_names[2])
	data_processor = None
	config_kwargs = {
		"output_dir": f"./temp/ppo+{model_name_or_path.split('/')[-1]}+{dataset_name.split('/')[-1]}",
		"model_name_or_path": model_name_or_path,
		"dataset_name": dataset_name,
		"reward_model_path": reward_model_path,
		"trust_remote_code": True,
		"dataset_train_split": "train[:500]",
		"dataset_test_split": "validation[:100]",
		"use_peft": True,
		"report_to": "none",
		"lora_target_modules": ["query_key_value"],
	}
	trainer_kwargs = {
	}
	ppo_pipeline(data_processor, config_kwargs, trainer_kwargs)

def dpo_pipeline_test():
	logging.info("DPO unittest ...")
	model_name_or_path = os.path.join(model_home, model_names[0])
	dataset_name = os.path.join(dataset_home, dataset_names[2])
	data_processor = None
	config_kwargs = {
		"output_dir": f"./temp/dpo+{model_name_or_path.split('/')[-1]}+{dataset_name.split('/')[-1]}",
		"model_name_or_path": model_name_or_path,
		"dataset_name": dataset_name,
		"trust_remote_code": True,
		"dataset_train_split": "descriptiveness[:500]",
		"dataset_test_split": "descriptiveness[500:600]",
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
		"use_peft": True,
		"report_to": "none",
		"lora_target_modules": ["q_proj", "k_proj", "v_proj"]
	}
	trainer_kwargs = {
		"reward_funcs": reward_funcs,
	}
	grpo_pipeline(data_processor, config_kwargs, trainer_kwargs)
	