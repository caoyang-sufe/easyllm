# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@stu.sufe.edu.cn

import platform

if platform.system() == "Linux":
	evaluate_home = "/nfsshare/home/caoyang/resource/evaluate"
	model_home = "/nfsshare/home/caoyang/resource/model"
	dataset_home = "/nfsshare/home/caoyang/resource/dataset"
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
	]

elif platform.system() == "Windows":
	evaluate_home = r"D:\resource\evaluate"
	model_home = r"D:\resource\model\huggingface"
	dataset_home = r"D:\resource\data\huggingface"
	model_names = [
		r"Qwen\Qwen2.5-0.5B-Instruct",	# 0
		r"EleutherAI\pythia-1b-deduped",	# 1
		r"EleutherAI\pythia-160m",	# 2
		r"trl-lib\Qwen2-0.5B-Reward",	# 3
		r"deepseek-ai\DeepSeek-R1-Distill-Qwen-1.5B",	# 4
		r"deepseek-ai\DeepSeek-R1-Distill-Qwen-32B",	# 5
		r"deepseek-ai\deepseek-math-7b-base",	# 6
		r"deepseek-ai\deepseek-moe-16b-base",	# 7 `trust_remote`
		r"Qwen\Qwen1.5-7B",	# 8
		r"Qwen\Qwen2.5-7B-Instruct",	# 9
		r"meta-llama\llama-2-7b-hf",	# 10
		r"meta-llama\Meta-Llama-3.1-8B-Instruct-hf",	# 11
		r"Qwen\Qwen3-8B-Instruct", 	# 12
		r"Qwen\Qwen3-0.6B", # 13
	]
	dataset_names = [
		r"trl-lib\tldr",	# 0 train["prompt", "completion"] + validation["prompt", "completion"] + test["prompt", "completion"]
		r"trl-lib\ultrafeedback_binarized",	# 1 train["chosen", "rejected", "score_chosen", "score_rejected"] + test["chosen", "rejected", "score_chosen", "score_rejected"]
		r"trl-internal-testing\descriptiveness-sentiment-trl-style", # 2 sentiment["prompt", "chosen", "rejected"] + descriptiveness["prompt", "chosen", "rejected"]
		r"YeungNLP\firefly-train-1.1M", # 3 train["input", "target"]
		r"openai\gsm8k",	 # 4 train["question", "answer"] + test["question", "answer"]
		r"HuggingFaceH4\MATH-500",	# 5 test["problem", "answer"]
		r"newfacade\LeetCodeDataset",	# 6 train["query", "response"] + test["query", "response"]
		r"larryvrh\Chinese-Poems", # 7 train["content"], you need to manually split
		r"HuggingFaceH4\MATH", # 8 train["problem", "solution"] + test["problem", "solution"]
	]

else:
	raise Exception(f"Unknown system: {platform.system()}")

LONG_PROMPT = [
]

def easy_unittest():
	# Test if the order to load different adapters influences the model parameters value
	import torch
	import logging
	from peft import PeftModel
	from transformers import AutoModelForCausalLM
	device = "cpu"
	base_model_path = "/nfsshare/home/caoyang/resource/model/Qwen/Qwen3-8B-Instruct"
	# Model 1
	output_dir_1 = "/nfsshare/home/caoyang/caoyang/easyllm/temp/sft-7b/sft+Qwen3-8B-Instruct+MATH-500+20250924074338"
	model_1 = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="auto")
	model_1 = PeftModel.from_pretrained(model_1, output_dir_1)
	model_1 = model_1.merge_and_unload()
	model_1 = PeftModel.from_pretrained(model_1, output_dir_2)
	model_1 = model_1.merge_and_unload()
	# Model 2
	output_dir_2 = "/nfsshare/home/caoyang/caoyang/easyllm/temp/sft+Qwen3-8B-Instruct+gsm8k+20251007220921"
	model_2 = AutoModelForCausalLM.from_pretrained(base_model_path, device_map="auto")
	model_2 = PeftModel.from_pretrained(model_2, output_dir_2)
	model_2 = model_2.merge_and_unload()
	model_2 = PeftModel.from_pretrained(model_2, output_dir_1)
	model_2 = model_2.merge_and_unload()

	for (name_1, parameter_1), (name_2, parameter_2) in zip(model_1.named_parameters(), model_2.named_parameters()):
		assert name_1 == name_2, f"Different paramter_name: {name_1} & {name_2}\n{parameter_1}\n----\n{parameter_2}"
		if torch.allclose(parameter_1, parameter_2, rtol=1e-5):
			print(f"Parameter {name_1} is all close")
			logging.info(f"Parameter {name_1} is all close")
		else:
			print(f"Parameter {name_1} is not all close")
			logging.info(f"Parameter {name_1} is not all close")
