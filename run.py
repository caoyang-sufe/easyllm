# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@stu.sufe.edu.cn

import os
import time
import torch
import logging
import argparse

from src.tools.easy import initialize_logger, terminate_logger
from src.unittests.trainer_pipelines import (
	sft_pipeline_test,
	dpo_pipeline_test,
	grpo_pipeline_test,
	ppo_pipeline_test,
)
from src.unittests.generate_pipelines import (
	decode_pipeline_test,
	generate_pipeline_test,
	one_time_forward_pipeline_test,
)
from src.unittests.analysis_pipelines import (
	skip_layer_generation_test_1,
	skip_layer_generation_test_2
)
from src.unittests.evaluator_pipelines import (
	evaluate_math_500,
	evaluate_gsm8k,
	evaluate_leetcodedataset,
	evaluate_chinese_poems,
)
from src.unittests import easy_unittest

os.makedirs("./log", exist_ok=True)
os.makedirs("./temp", exist_ok=True)
with open("check.txt", 'w', encoding="utf8") as f:
	f.write(f"{torch.cuda.is_available()}\n")
	f.write(f"{torch.backends.mps.is_available()}\n")
	f.write(f"{torch.cuda.device_count()}\n")


########################################################################
# ----------------------------------------------------------------------
# FUNCTION NAME
# ----------------------------------------------------------------------
# 1. TRAINER
# ----------------------------------------------------------------------
function_name = "sft_pipeline_test"
# function_name = "dpo_pipeline_test"
# function_name = "grpo_pipeline_test"
# function_name = "ppo_pipeline_test"
# ----------------------------------------------------------------------
# 2. EVALUATOR
# ----------------------------------------------------------------------
# function_name = "evaluate_math_500"
# function_name = "evaluate_gsm8k"
# function_name = "evaluate_leetcodedataset"
# function_name = "evaluate_chinese_poems"
# ----------------------------------------------------------------------
# 3. GENERATE
# ----------------------------------------------------------------------
# function_name = "one_time_forward_pipeline_test"
# function_name = "decode_pipeline_test"
# function_name = "generate_pipeline_test"
# ----------------------------------------------------------------------
# 4. ANALYSIS
# ----------------------------------------------------------------------
# function_name = "skip_layer_generation_test_1"
# function_name = "skip_layer_generation_test_2"
# ----------------------------------------------------------------------
# END
# ----------------------------------------------------------------------
########################################################################

if "function_name" in dir():
	logger = initialize_logger(f"./log/{function_name}+{time.strftime('%Y-%m-%d-%H-%M-%S')}.log", mode='w')
else:
	parser = argparse.ArgumentParser("--")
	parser.add_argument("--name", default="2-stage-sft-from-chinese-math", type=str)	# e.g. "esg_crawler", "esg_downloader", "csdn_watcher_and_reader"
	args = parser.parse_args()
	logger = initialize_logger(f"./log/{args.name}+{time.strftime('%Y-%m-%d-%H-%M-%S')}.log", mode='w')
########################################################################
# ----------------------------------------------------------------------
# FUNCTION CALL
# ----------------------------------------------------------------------
# 1. TRAINER
# ----------------------------------------------------------------------

## 1.1 1-stage-sft
# model_ids = [11, 10, 9, 8, 12]	# 7B-level model
# train_dataset_ids = [4, 6, 7, 8]	# ! MATH-500, MATH-Chinese
# logger.info(f"model_ids: {model_ids} - train_dataset_ids: {train_dataset_ids}")
# for train_dataset_id in train_dataset_ids:
	# for model_id in model_ids:
		# logger.info(f"model_id: {model_id} - train_dataset_id: {train_dataset_id}")
		# sft_pipeline_test(
			# model_id = model_id,
			# train_dataset_id = train_dataset_id,
			# eval_dataset_ids = [train_dataset_id],
			# adapter_output_dirs = None,
			# per_device_train_batch_size = 8,
			# per_device_eval_batch_size = 8,
			# num_train_epochs = 16,
			# n_cuda = 2,
			# use_overwritten_model_class = True,
			# experiment_name = "7b-model-1-stage-sft",
		# )

## 1.2 2-stage-sft
model_names = [
	"Qwen/Qwen2.5-0.5B-Instruct",	# 0
	"EleutherAI/pythia-1b-deduped",	# 1
	"EleutherAI/pythia-160m",	# 2
	"trl-lib/Qwen2-0.5B-Reward",	# 3
	"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",	# 4
	"deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",	# 5
	"deepseek-ai/deepseek-math-7b-base",	# 6
	"deepseek-ai/deepseek-moe-16b-base",	# 7 `trust_remote`
	"Qwen/Qwen1.5-7B",	# 8
	"Qwen/Qwen2.5-7B-Instruct",	# 9
	"meta-llama/llama-2-7b-hf",	# 10
	"meta-llama/Meta-Llama-3.1-8B-Instruct-hf",	# 11
	"Qwen/Qwen3-8B-Instruct", 	# 12
	"Qwen/Qwen3-0.6B", # 13
]
save_dir = "/nfsshare/home/caoyang/caoyang/easyllm/temp/1-stage-sft/2-for-base/"
model_ids_to_adapter_output_dirs = {
	8: {
		5: os.path.join(save_dir, "sft+Qwen1.5-7B+MATH-500+20251023184858/checkpoint-250"),
		9: os.path.join(save_dir, "sft+Qwen1.5-7B+Math-Chinese-DeepSeek-R1-10K+20251024001921/checkpoint-1500"),
	},
	9: {
		5: os.path.join(save_dir, "sft+Qwen2.5-7B-Instruct+MATH-500+20251023183040/checkpoint-800"),
		9: os.path.join(save_dir, "sft+Qwen2.5-7B-Instruct+Math-Chinese-DeepSeek-R1-10K+20251023224914/checkpoint-2000"),
	},
	10: {
		5: os.path.join(save_dir, "sft+llama-2-7b-hf+MATH-500+20251023180939/checkpoint-300"),
		9: os.path.join(save_dir, "sft+llama-2-7b-hf+Math-Chinese-DeepSeek-R1-10K+20251021182355/checkpoint-2000"),
	},
	11: {
		5: os.path.join(save_dir, "sft+Meta-Llama-3.1-8B-Instruct-hf+MATH-500+20251023175011/checkpoint-350"),
		9: os.path.join(save_dir, "sft+Meta-Llama-3.1-8B-Instruct-hf+Math-Chinese-DeepSeek-R1-10K+20251023192050/checkpoint-1875"),
	},
	12: {
		5: os.path.join(save_dir, "sft+Qwen3-8B-Instruct+MATH-500+20251023190810/checkpoint-800"),
		9: os.path.join(save_dir, "sft+Qwen3-8B-Instruct+Math-Chinese-DeepSeek-R1-10K+20251024015850/checkpoint-2000"),
	},
}
dataset_names = [
	"trl-lib/tldr",	# 0 train["prompt", "completion"] + validation["prompt", "completion"] + test["prompt", "completion"]
	"trl-lib/ultrafeedback_binarized",	# 1 train["chosen", "rejected", "score_chosen", "score_rejected"] + test["chosen", "rejected", "score_chosen", "score_rejected"]
	"trl-internal-testing/descriptiveness-sentiment-trl-style", # 2 sentiment["prompt", "chosen", "rejected"] + descriptiveness["prompt", "chosen", "rejected"]
	"YeungNLP/firefly-train-1.1M", # 3 train["input", "target"]
	"openai/gsm8k",	# 4 train["question", "answer"] + test["question", "answer"]
	"HuggingFaceH4/MATH-500",	# 5 test["problem", "answer"]
	"newfacade/LeetCodeDataset",	# 6 train["query", "response"] + test["query", "response"]
	"larryvrh/Chinese-Poems", # 7 train["content"], you need to manually split
	"HuggingFaceH4/MATH", # 8 train["problem", "solution"] + test["problem", "solution"]
	"MxMode/Math-Chinese-DeepSeek-R1-10K",	# 9 train["prompt", "reasoning", "response"]
]
model_ids = [10, 11, 12, 9, 8]
model_ids = [12, 9, 8]
model_ids = [8]
train_dataset_ids = [6, 7, 8, 9, 4]
eval_dataset_id = 5	# MATH-500
logger.info(f"model_ids: {model_ids} - train_dataset_ids: {train_dataset_ids} - eval_dataset_id: {eval_dataset_id}")
for model_id in model_ids:
	for train_dataset_id in train_dataset_ids:
		logger.info(f"model_id: {model_id} - train_dataset_id: {train_dataset_id}")
		sft_pipeline_test(
			model_id = model_id,
			train_dataset_id = train_dataset_id,
			eval_dataset_ids = [train_dataset_id, eval_dataset_id],
			adapter_output_dirs = [model_ids_to_adapter_output_dirs[model_id][eval_dataset_id]],
			per_device_train_batch_size = 8,
			per_device_eval_batch_size = 8,
			num_train_epochs = 32,
			n_cuda = 2,
			use_overwritten_model_class = True,
			experiment_name = "7b-models-2-stage-sft-based-on-MATH-500"
		)
# ----------------------------------------------------------------------
# 2. EVALUATOR
# ----------------------------------------------------------------------
# eval(function_name)(
	# model_id = 10,
	# overwritten_model_class = "ParallelLlamaForCausalLM",
	# n_cuda = 2,
	# do_sample = False,
	# adapter_output_dirs = [
		# "/nfsshare/home/caoyang/caoyang/easyllm/temp/1-stage-sft/sft+llama-2-7b-hf+MATH-500+20250924023919",
		# "/nfsshare/home/caoyang/caoyang/easyllm/temp/2-stage-sft/sft+llama-2-7b-hf+LeetCodeDataset+20251012171429",
	# ],
# )
# eval(function_name)(
	# model_id = 10,
	# overwritten_model_class = "ParallelLlamaForCausalLM",
	# n_cuda = 2,
	# do_sample = False,
	# adapter_output_dirs = [
		# "/nfsshare/home/caoyang/caoyang/easyllm/temp/1-stage-sft/sft+llama-2-7b-hf+MATH-500+20250924023919",
		# "/nfsshare/home/caoyang/caoyang/easyllm/temp/2-stage-sft/sft+llama-2-7b-hf+LeetCodeDataset+20251014163052",
	# ],
# )
# eval(function_name)(
	# model_id = 10,
	# overwritten_model_class = "ParallelLlamaForCausalLM",
	# n_cuda = 2,
	# do_sample = False,
	# adapter_output_dirs = [
		# "/nfsshare/home/caoyang/caoyang/easyllm/temp/1-stage-sft/sft+llama-2-7b-hf+MATH-500+20250924023919",
		# "/nfsshare/home/caoyang/caoyang/easyllm/temp/2-stage-sft/sft+llama-2-7b-hf+LeetCodeDataset+20251015003243",
	# ],
# )
# eval(function_name)(
	# model_id = 12,
	# overwritten_model_class = "ParallelQwen3ForCausalLM",
	# n_cuda = 2,
	# do_sample = False,
	# adapter_output_dirs = [
		# "/nfsshare/home/caoyang/caoyang/easyllm/temp/1-stage-sft/sft+Qwen3-8B-Instruct+MATH-500+20250924074338",
		# "/nfsshare/home/caoyang/caoyang/easyllm/temp/2-stage-sft/sft+Qwen3-8B-Instruct+LeetCodeDataset+20251011213455",
	# ],
# )
# eval(function_name)(
	# model_id = 12,
	# overwritten_model_class = "ParallelQwen3ForCausalLM",
	# n_cuda = 2,
	# do_sample = False,
	# adapter_output_dirs = [
		# "/nfsshare/home/caoyang/caoyang/easyllm/temp/1-stage-sft/sft+Qwen3-8B-Instruct+MATH-500+20250924074338",
		# "/nfsshare/home/caoyang/caoyang/easyllm/temp/2-stage-sft/sft+Qwen3-8B-Instruct+LeetCodeDataset+20251012040852",
	# ],
# )
# eval(function_name)(
	# model_id = 12,
	# overwritten_model_class = "ParallelQwen3ForCausalLM",
	# n_cuda = 2,
	# do_sample = False,
	# adapter_output_dirs = [
		# "/nfsshare/home/caoyang/caoyang/easyllm/temp/1-stage-sft/sft+Qwen3-8B-Instruct+MATH-500+20250924074338",
		# "/nfsshare/home/caoyang/caoyang/easyllm/temp/2-stage-sft/sft+Qwen3-8B-Instruct+LeetCodeDataset+20251012104206",
	# ],
# )
# ----------------------------------------------------------------------
# 3. GENERATE
# ----------------------------------------------------------------------
# one_time_forward_pipeline_test(model_id=-1, device=None, overwritten_model_class=None, n_cuda=2, s=0)
# decode_pipeline_test(model_id=10, device="cpu")
# decode_pipeline_test(model_id=11, device="cpu")
# generate_pipeline_test(model_id=-1, device=None)
# ----------------------------------------------------------------------
# 4. ANALYSIS
# ----------------------------------------------------------------------
# eval(function_name)(model_id = 0, device="cuda")
# ----------------------------------------------------------------------
# END
# ----------------------------------------------------------------------
########################################################################
terminate_logger(logger)
########################################################################
# ----------------------------------------------------------------------
model_names = [
	"Qwen/Qwen2.5-0.5B-Instruct",	# 0
	"EleutherAI/pythia-1b-deduped",	# 1
	"EleutherAI/pythia-160m",	# 2
	"trl-lib/Qwen2-0.5B-Reward",	# 3
	"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",	# 4
	"deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",	# 5
	"deepseek-ai/deepseek-math-7b-base",	# 6
	"deepseek-ai/deepseek-moe-16b-base",	# 7 `trust_remote`
	"Qwen/Qwen1.5-7B",	# 8
	"Qwen/Qwen2.5-7B-Instruct",	# 9
	"meta-llama/llama-2-7b-hf",	# 10
	"meta-llama/Meta-Llama-3.1-8B-Instruct-hf",	# 11
	"Qwen/Qwen3-8B-Instruct", 	# 12
	"Qwen/Qwen3-0.6B", # 13
]
dataset_names = [
	"trl-lib/tldr",	# 0 train["prompt", "completion"] + validation["prompt", "completion"] + test["prompt", "completion"]
	"trl-lib/ultrafeedback_binarized",	# 1 train["chosen", "rejected", "score_chosen", "score_rejected"] + test["chosen", "rejected", "score_chosen", "score_rejected"]
	"trl-internal-testing/descriptiveness-sentiment-trl-style", # 2 sentiment["prompt", "chosen", "rejected"] + descriptiveness["prompt", "chosen", "rejected"]
	"YeungNLP/firefly-train-1.1M", # 3 train["input", "target"]
	"openai/gsm8k",	# 4 train["question", "answer"] + test["question", "answer"]
	"HuggingFaceH4/MATH-500",	# 5 test["problem", "answer"]
	"newfacade/LeetCodeDataset",	# 6 train["query", "response"] + test["query", "response"]
	"larryvrh/Chinese-Poems", # 7 train["content"], you need to manually split
	"HuggingFaceH4/MATH", # 8 train["problem", "solution"] + test["problem", "solution"]
	"MxMode/Math-Chinese-DeepSeek-R1-10K",	# 9 train["prompt", "reasoning", "response"]
]
