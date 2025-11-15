# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@stu.sufe.edu.cn

import platform

model_parallel_classes_map = {
	0: None,	# Small model
	1: None,	# NotImplemented
	2: None,	# NotImplemented
	3: None,	# NotImplemented
	4: None,	# Small model
	5: "ParallelQwen2ForCausalLM",
	6: "ParallelLlamaForCausalLM",
	7: "ParallelDeepseekForCausalLM",
	8: "ParallelQwen2ForCausalLM",
	9: "ParallelQwen2ForCausalLM",
	10: "ParallelLlamaForCausalLM",
	11: "ParallelLlamaForCausalLM",
	12: "ParallelQwen3ForCausalLM",
	13: None,	# Small model
}

dataset_processors_map = {
	0: {"train": lambda _x: {"prompt": _x["prompt"], "completion": _x["completion"]}, "test": lambda _x: {"prompt": _x["prompt"], "completion": _x["completion"]}},	# tldr
	1: None,	# DPO Not Implemented
	2: None,	# DPO Not Implemented
	3: {"train": lambda _x: {"prompt": _x["input"], "completion": _x["target"]}, "test": lambda _x: {"prompt": _x["input"], "completion": _x["target"]}},	# firefly
	4: {"train": lambda _x: {"prompt": _x["question"], "completion": _x["answer"]}, "test": lambda _x: {"prompt": _x["question"], "completion": _x["answer"]}},	# gsm8k
	5: {"train": lambda _x: {"prompt": _x["problem"], "completion": _x["answer"]}, "test": lambda _x: {"prompt": _x["problem"], "completion": _x["answer"]}},	# MATH-500
	6: {"train": lambda _x: {"prompt": _x["query"], "completion": _x["response"]}, "test": lambda _x: {"prompt": _x["query"], "completion": _x["response"]}},	# LeetCodeDataset
	7: {"train": lambda _x: {"prompt": _x["content"][:-10], "completion": _x["content"][-10:]}, "test": lambda _x: {"prompt": _x["content"][:-10], "completion": _x["content"][-10:]}},	# Chinese-Poems
	8: {"train": lambda _x: {"prompt": _x["problem"], "completion": _x["solution"]}, "test": lambda _x: {"prompt": _x["problem"], "completion": _x["solution"]}},	# MATH
	9: {"train": lambda _x: {"prompt": _x["prompt"], "completion": _x["response"]}, "test": lambda _x: {"prompt": _x["prompt"], "completion": _x["response"]}},	# Math-Chinese-DeepSeek-R1-10K
}

dataset_train_test_splits_map = {
	0: {"train": "train[:1000]", "test": "validation[:200]"},	# tldr
	1: None,	# DPO Not Implemented
	2: None,	# DPO Not Implemented
	3: {"train": "train[:1000]", "test": "train[1000:1100]"},	# firefly
	4: {"train": "train", "test": "test"},	# gsm8k
	5: {"train": "test[:400]", "test": "test[400:]"},	# MATH-500
	6: {"train": "train", "test": "test"},	# LeetCodeDataset
	7: {"train": "train[:1000]", "test": "train[1000:1100]"},	# Chinese-Poems
	8: {"train": "train", "test": "test"},	# MATH
	9: {"train": "train[:1000]", "test": "train[1000:1100]"},	# Math-Chinese-DeepSeek-R1-10K
}

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
		"openai/gsm8k",	# 4 train7473["question", "answer"] + test1319["question", "answer"]
		"HuggingFaceH4/MATH-500",	# 5 test500["problem", "answer"]
		"newfacade/LeetCodeDataset",	# 6 train2461["query", "response"] + test228["query", "response"]
		"larryvrh/Chinese-Poems", # 7 train["content"], you need to manually split
		"HuggingFaceH4/MATH", # 8 train["problem", "solution"] + test["problem", "solution"]
		"MxMode/Math-Chinese-DeepSeek-R1-10K",	# 9 train["prompt", "reasoning", "response"]
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
		r"MxMode\Math-Chinese-DeepSeek-R1-10K",	# 9 train["prompt", "reasoning", "response"]
	]

else:
	raise Exception(f"Unknown system: {platform.system()}")

LONG_PROMPT = [
]

# Test if the order to load different adapters influences the model parameters value
def _unittest_1():
	
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

# Test different plots
def _unittest_2():
	fig_test, axes_test = plt.subplots(2, 2, figsize=(10, 8))
	# 1 Line plot
	x = np.linspace(0, 10, 100)
	axes_test[0, 0].plot(x, np.log(x), 'r-', label='log(x, e)')
	axes_test[0, 0].plot(x, np.log2(x), 'r-', label='log(x, 2)')
	axes_test[0, 0].plot(x, np.log10(x), 'r-', label='log(x, 10)')
	axes_test[0, 0].set_title('Line Plot')
	axes_test[0, 0].legend()
	# 2 Scatter plot
	x_scatter = np.random.randn(50)
	y_scatter = np.random.randn(50)
	colors = np.random.rand(50)
	axes_test[0, 1].scatter(x_scatter, y_scatter, c=colors, cmap='viridis')
	axes_test[0, 1].set_title('Scatter Plot')
	# 3 Bar plot
	categories = ['A', 'B', 'C', 'D']
	values = [5, 7, 3, 8]
	axes_test[1, 0].bar(categories, values, alpha=0.7)
	axes_test[1, 0].set_title('Bar Chart')
	# 4 Histogram plot
	data_hist = np.random.randn(1000)
	axes_test[1, 1].hist(data_hist, bins=30, alpha=0.7)
	axes_test[1, 1].set_title('Histogram')
	# 5 Heatmap plot
	# heatmap_kwargs = {
	#     "cmap": "binary",
	#     "annot": False,
	#     "fmt": ".2f",
	#     "cbar": True,
	# }
	# import seaborn as sns
	# data = np.random.random((10, 10))
	# sns.heatmap(data, ax=axes_test[1, 1], **heatmap_kwargs)
	plt.tight_layout()
	plt.show()
