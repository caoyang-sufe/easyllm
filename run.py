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
	sft_train_math_500,
	sft_train_gsm8k,
	sft_train_leetcodedataset,
	sft_train_chinese_poems,
	sft_train_math,
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
# function_name = "sft_pipeline_test"
# function_name = "sft_train_math_500"
# function_name = "sft_train_gsm8k"
# function_name = "sft_train_leetcodedataset"
# function_name = "sft_train_chinese_poems"
function_name = "sft_train_math"
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
logger = initialize_logger(f"./log/{function_name}+{time.strftime('%Y-%m-%d-%H-%M-%S')}.log", mode='w')
########################################################################
# ----------------------------------------------------------------------
# FUNCTION CALL
# ----------------------------------------------------------------------
# 1. TRAINER
# ----------------------------------------------------------------------

sft_train_chinese_poems(
	model_id = 0,
	overwritten_model_class = None,
	n_cuda = 0,
	adapter_output_dirs = None,
	debug = True,
)

# eval(function_name)(
	# model_id = 12,
	# overwritten_model_class = "ParallelQwen3ForCausalLM",
	# n_cuda = 2,
	# adapter_output_dirs = ["/nfsshare/home/caoyang/caoyang/easyllm/temp/1-stage-sft/sft+Qwen3-8B-Instruct+MATH-500+20250924074338"],
# )

# eval(function_name)(
	# model_id = 10,
	# overwritten_model_class = "ParallelLlamaForCausalLM",
	# n_cuda = 2,
	# adapter_output_dirs = ["/nfsshare/home/caoyang/caoyang/easyllm/temp/1-stage-sft/sft+llama-2-7b-hf+MATH-500+20250924023919"],
# )

# sft_train_leetcodedataset(
	# model_id = 12,
	# overwritten_model_class = "ParallelQwen3ForCausalLM",
	# n_cuda = 2,
	# adapter_output_dirs = ["/nfsshare/home/caoyang/caoyang/easyllm/temp/1-stage-sft/sft+Qwen3-8B-Instruct+MATH-500+20250924074338"],
# )

# sft_train_gsm8k(
	# model_id = 12,
	# overwritten_model_class = "ParallelQwen3ForCausalLM",
	# n_cuda = 2,
	# adapter_output_dirs = ["/nfsshare/home/caoyang/caoyang/easyllm/temp/1-stage-sft/sft+Qwen3-8B-Instruct+MATH-500+20250924074338"],
# )

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
# decode_pipeline_test(model_id=-1, device=None)
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
	"openai/gsm8k",	 # 4
	"HuggingFaceH4/MATH-500",	# 5
	"newfacade/LeetCodeDataset",	# 6
]
