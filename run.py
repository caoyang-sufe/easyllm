# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@stu.sufe.edu.cn

import os
import time
import torch
import pandas
import logging
from datasets import load_dataset
from torch.nn import functional as F
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from src.tools.accelerate import accelerate_load_model
from src.tools.easy import initialize_logger, terminate_logger
from src.pipelines.trainer import base_pipeline, sft_pipeline, ppo_pipeline, dpo_pipeline, grpo_pipeline
from src.tools.transformers import greedy_decode, k_step_greedy_decode, beam_search_decode, generate_token_prob
from src.unittests.trainer_pipelines import sft_pipeline_test, dpo_pipeline_test, grpo_pipeline_test, ppo_pipeline_test, sft_train_math_500, sft_train_gsm8k
from src.unittests.generate_pipelines import decode_pipeline_test, generate_pipeline_test
from src.unittests.analysis_pipelines import skip_layer_generation_test_1, skip_layer_generation_test_2
from src.unittests.evaluator_pipelines import evaluate_math_500, evaluate_gsm8k

os.makedirs("./log", exist_ok=True)
os.makedirs("./temp", exist_ok=True)

with open("check.txt", 'w', encoding="utf8") as f:
	f.write(f"{torch.cuda.is_available()}\n")
	f.write(f"{torch.backends.mps.is_available()}\n")
	f.write(f"{torch.cuda.device_count()}\n")

# function_name = "sft_pipeline_test"
# function_name = "sft_train_math_500"
# function_name = "sft_train_gsm8k"
# function_name = "dpo_pipeline_test"
# function_name = "grpo_pipeline_test"
# function_name = "ppo_pipeline_test"
function_name = "decode_pipeline_test"

# function_name = "generate_pipeline_test"
# function_name = "skip_layer_generation_test_1"
# function_name = "skip_layer_generation_test_2"

# function_name = "evaluate_math_500"
# function_name = "evaluate_gsm8k"
logger = initialize_logger(f"./log/{function_name}+{time.strftime('%Y-%m-%d-%H-%M-%S')}.log", mode='w')
# ----------------------------------------------------------------------

eval(function_name)()

# eval(function_name)(model_id=9, device="cuda")	# skip_layer_generation_test_X
# eval(function_name)(model_id=10, device="cuda")	# skip_layer_generation_test_X
# eval(function_name)(model_id=11, device="cuda")	# skip_layer_generation_test_X
# eval(function_name)(model_id=12, device="cuda")	# skip_layer_generation_test_X
# eval(function_name)(model_id=0, parallel_model_class=None, n_cuda=2)
# eval(function_name)(model_id=9, parallel_model_class="ParallelQwen2ForCausalLM", n_cuda=2)


# eval(function_name)(model_id=13, parallel_model_class=None, n_cuda=2)
# eval(function_name)(model_id=11, parallel_model_class="ParallelLlamaForCausalLM", n_cuda=2)
# eval(function_name)(model_id=10, parallel_model_class="ParallelLlamaForCausalLM", n_cuda=2)
# eval(function_name)(model_id=9, parallel_model_class="ParallelQwen2ForCausalLM", n_cuda=2)
# eval(function_name)(model_id=12, parallel_model_class="ParallelQwen3ForCausalLM", n_cuda=2)
# eval(function_name)(model_id=9, parallel_model_class="ParallelQwen2ForCausalLM", n_cuda=2)

# eval(function_name)(model_id=0, parallel_model_class=None)
# eval(function_name)(model_id=9, parallel_model_class="ParallelQwen2ForCausalLM", n_cuda=2)
# eval(function_name)(model_id=12, parallel_model_class="ParallelQwen3ForCausalLM", n_cuda=2)
# eval(function_name)(model_id=10, parallel_model_class="ParallelLlamaForCausalLM", n_cuda=2)
# eval(function_name)(model_id=11, parallel_model_class="ParallelLlamaForCausalLM", n_cuda=2)

terminate_logger(logger)



# model_names = [
	# "Qwen/Qwen2.5-0.5B-Instruct",	# 0
	# "EleutherAI/pythia-1b-deduped",	# 1
	# "EleutherAI/pythia-160m",	# 2
	# "trl-lib/Qwen2-0.5B-Reward",	# 3
	# "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",	# 4
	# "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",	# 5
	# "deepseek-ai/deepseek-math-7b-base",	# 6
	# "deepseek-ai/deepseek-moe-16b-base",	# 7 `trust_remote`
	# "Qwen/Qwen1.5-7B",	# 8
	# "Qwen/Qwen2.5-7B-Instruct",	# 9
	# "meta-llama/llama-2-7b-hf",	# 10
	# "meta-llama/Meta-Llama-3.1-8B-Instruct-hf",	# 11
	# "Qwen/Qwen3-8B-Instruct", 	# 12
	# "Qwen/Qwen3-0.6B", # 13
# ]

