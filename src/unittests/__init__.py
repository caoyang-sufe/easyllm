# -*- coding: utf8 -*-
# @author: caoyang
# @email: caoyang@stu.sufe.edu.cn

import platform



if platform.system() == "Linux":
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
		"Qwen/Qwen2.5-7B-Instruct",
	]
	dataset_names = [
		"trl-lib/tldr",	# train["prompt", "completion"] + validation["prompt", "completion"] + test["prompt", "completion"]
		"trl-lib/ultrafeedback_binarized",	# train["chosen", "rejected", "score_chosen", "score_rejected"] + test["chosen", "rejected", "score_chosen", "score_rejected"]
		"trl-internal-testing/descriptiveness-sentiment-trl-style", # sentiment["prompt", "chosen", "rejected"] + descriptiveness["prompt", "chosen", "rejected"]
		"YeungNLP/firefly-train-1.1M", # train["input", "target"]
	]

elif platform.system() == "Windows":
	model_home = r"D:\resource\model\huggingface"
	dataset_home = r"D:\resource\data\huggingface"
	model_names = [
		r"Qwen\Qwen2.5-0.5B-Instruct",
		r"EleutherAI\pythia-1b-deduped",
		r"EleutherAI\pythia-160m",
		r"trl-lib\Qwen2-0.5B-Reward",
		r"deepseek-ai\DeepSeek-R1-Distill-Qwen-1.5B",
		r"deepseek-ai\DeepSeek-R1-Distill-Qwen-32B",
		r"deepseek-ai\deepseek-math-7b-base",
		r"deepseek-ai\deepseek-moe-16b-base",	# `trust_remote`
		r"Qwen\Qwen1.5-7B",
		r"Qwen\Qwen2.5-7B-Instruct",
	]
	dataset_names = [
		r"trl-lib\tldr",	# train["prompt", "completion"] + validation["prompt", "completion"] + test["prompt", "completion"]
		r"trl-lib\ultrafeedback_binarized",	# train["chosen", "rejected", "score_chosen", "score_rejected"] + test["chosen", "rejected", "score_chosen", "score_rejected"]
		r"trl-internal-testing\descriptiveness-sentiment-trl-style", # sentiment["prompt", "chosen", "rejected"] + descriptiveness["prompt", "chosen", "rejected"]
		r"YeungNLP\firefly-train-1.1M", # train["input", "target"]
	]
	
else:
	raise Exception(f"Unknown system: {platform.system()}")
	
