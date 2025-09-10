# -*- coding: utf8 -*-
# @author: caoyang
# @email: caoyang@stu.sufe.edu.cn

model_home = "/nfsshare/home/caoyang/resource/model"
dataset_home = "/nfsshare/home/caoyang/resource/dataset"
model_names = [
	"Qwen/Qwen2.5-0.5B-Instruct",
	"EleutherAI/pythia-1b-deduped",
	"EleutherAI/pythia-160m",
	"trl-lib/Qwen2-0.5B-Reward",
	"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
	"deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
	"deepseek-ai/deepseek-math-7b-base",
	"deepseek-ai/deepseek-moe-16b-base",	# `trust_remote`
	"Qwen/Qwen1.5-7B",
]

dataset_names = [
	"trl-lib/tldr",	# train["prompt", "completion"] + validation["prompt", "completion"] + test["prompt", "completion"]
	"trl-lib/ultrafeedback_binarized",	# train["chosen", "rejected", "score_chosen", "score_rejected"] + test["chosen", "rejected", "score_chosen", "score_rejected"]
	"trl-internal-testing/descriptiveness-sentiment-trl-style", # sentiment["prompt", "chosen", "rejected"] + descriptiveness["prompt", "chosen", "rejected"]
	"YeungNLP/firefly-train-1.1M", # train["input", "target"]
]
