# -*- coding: utf8 -*-
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
from src.unittests.trainer_pipelines import sft_pipeline_test, dpo_pipeline_test, grpo_pipeline_test, ppo_pipeline_test
from src.unittests.generate_pipelines import decode_pipeline_test, generate_pipeline_test
from src.unittests.analysis_pipelines import skip_layer_generation_test_1, skip_layer_generation_test_2

os.makedirs("./log", exist_ok=True)
os.makedirs("./temp", exist_ok=True)

with open("check.txt", 'w', encoding="utf8") as f:
	f.write(f"{torch.cuda.is_available()}\n")
	f.write(f"{torch.backends.mps.is_available()}\n")
	f.write(f"{torch.cuda.device_count()}\n")

function_name = "sft_pipeline_test"
# function_name = "dpo_pipeline_test"
# function_name = "grpo_pipeline_test"
# function_name = "ppo_pipeline_test"
# function_name = "decode_pipeline_test"
# function_name = "generate_pipeline_test"
# function_name = "skip_layer_generation_test_1"
# function_name = "skip_layer_generation_test_2"

logger = initialize_logger(f"./log/{function_name}+{time.strftime('%Y-%m-%d-%H-%M-%S')}.log", mode='w')
# eval(function_name)(model_id=8, device="cpu")	# skip_layer_generation_test_X
# eval(function_name)(model_id=9, device="cpu")	# skip_layer_generation_test_X
eval(function_name)()
terminate_logger(logger)
