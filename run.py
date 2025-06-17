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

MODEL_PATHS = [
	"/nfsshare/home/caoyang/resource/model/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
	"/nfsshare/home/caoyang/resource/model/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
	"/nfsshare/home/caoyang/resource/model/Qwen/Qwen1.5-0.5B",
	"/nfsshare/home/caoyang/resource/model/Qwen/Qwen1.5-0.5B-Chat",
	"/nfsshare/home/caoyang/resource/model/Qwen/Qwen2.5-0.5B-Instruct",
	"/nfsshare/home/caoyang/resource/model/Qwen/Qwen1.5-7B",
	"/nfsshare/home/caoyang/resource/model/Qwen/Qwen1.5-7B-Chat-GPTQ-Int4",
	"/nfsshare/home/caoyang/resource/model/Qwen/Qwen-7B-Chat",
]

os.makedirs("./log", exist_ok=True)
os.makedirs("./temp", exist_ok=True)

with open("check.txt", 'w', encoding="utf8") as f:
	f.write(f"{torch.cuda.is_available()}\n")
	f.write(f"{torch.backends.mps.is_available()}\n")
	f.write(f"{torch.cuda.device_count()}\n")

# Case Study

logger = initialize_logger(f"./log/run-{time.strftime('%Y-%m-%d-%H-%M-%S')}.log", mode='w')

# sft_pipeline_test()
# dpo_pipeline_test()
grpo_pipeline_test()
# ppo_pipeline_test()

terminate_logger(logger)
