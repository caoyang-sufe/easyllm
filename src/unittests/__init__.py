# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@stu.sufe.edu.cn

import platform

if platform.system() == "Linux":
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
		"openai/gsm8k",	 # 4 
		"HuggingFaceH4/MATH-500",	# 5
		"newfacade/LeetCodeDataset",	# 6
	]

elif platform.system() == "Windows":
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
		r"openai\gsm8k",	 # 4 
		r"HuggingFaceH4\MATH-500",	# 5
		r"newfacade\LeetCodeDataset",	# 6
	]
	
else:
	raise Exception(f"Unknown system: {platform.system()}")
	
LONG_PROMPT = [
# ----
r"""# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@stu.sufe.edu.cn
# Evaluator for CAUSAL_LM

import numpy
import torch
import logging
from copy import deepcopy
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.tools.transformers import generate_token_prob
from src.tools.metric import (
	calc_token_accuracy,
	calc_perplexity,
	calc_bleu,
	calc_rouge_n,
	calc_rouge_w,
)

# Base pipeline for evalution
# @param model: Huggingface AutoModelForCausalLM object
# @param tokenizer: Huggingface tokenizer object
# @param dataset: Huggingface dataset object, usually comes from a split of a whole dataset , e.g. dataset["test"]
# @param model_name_or_path: [Str] Either `model` or `model_name_or_path` is not None
# @param dataset_name_or_path: [Str] Either `dataset` or `dataset_name_or_path` is not None
# @param test_data_split: [Str] Take effect only when keyword argument `dataset` is None
# @param test_data_size: [Int|Float]
# - Default `None` means using the whole test dataset
# - You can set as a ratio (i.e. float in 0-1) or concrete number of test data
# @param device: [Str|torch.device]
# @param input_column: [Str] e.g. "prompt", "input", "text", "question", "problem", etc
# @param output_column: [Str] e.g. "completion", "output", "target", "answer", "response", etc
# @param do_sample: [Boolean]
# - Default `False` means to use greedy decode (i.e. only test once)
# - If set as `True`, then keyword argument `do_sample_times` and `do_sample_kwargs` will take effect
# @param do_sample_times: [Int] Number of sample times
# @param do_sample_kwargs: [Dict] Keyword arguments of `model.generate`, default `{"top_k": 0, "top_p": 1., "temperature": 1., "num_beams": 1}` referes to disable `top_k, top_p, temperature` and use greedy decode only
# @param use_cache: [Boolean] Keyword argument `use_cache` of `model.generate` (whether to use KV-Cache)
# @param metrics: [Tuple(Str, Dict, Str)]
#   - Str is the metric function name, e.g. "calc_perplexity", see `src.tools.metric` to find all functions
#   - Dict is the corresponding kwargs for the metric function functiontric
#   - Str is the metric name, e.g. "perplexity", it must be unique
#   - Currently support ["token_accuracy", "perplexity", "rouge_n", "rouge_w"],
def base_pipeline(model = None,
				  tokenizer = None,
				  dataset = None,
				  model_name_or_path = None,
				  dataset_name_or_path = None,
				  test_data_split = None,
				  test_data_size = None,
				  device = "cpu",
				  input_column = "prompt",
				  target_column = "completion",
				  do_sample = False,
				  do_sample_times = 10,
				  do_sample_kwargs = {"top_k": 0, "top_p": 1., "temperature": 1., "num_beams": 1},
				  use_cache = True,
				  input_max_length = 512,
				  target_max_length = 128,
				  metrics = [("calc_token_accuracy", {}, "token_accuracy"),
							 ("calc_perplexity", {}, "perplexity"),
							 ("calc_bleu", {"min_grams": 1, "max_grams": 3}, "bleu_3"),
							 ("calc_rouge_n", {'n': 3, "beta": 1}, "rouge_3"),
							 ("calc_rouge_w", {"weight_function": lambda _x: _x, "weight_function_reverse": lambda _x: _x, "beta": 1}, "rouge_l"),
							 ("calc_rouge_w", {"weight_function": lambda _x: _x ** 2, "weight_function_reverse": lambda _x: _x ** 0.5, "beta": 1}, "rouge_w"),
							 ],
				  ):
	if model is None:
		logging.info(f"Load model from {model_name_or_path}")
		model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)
	if tokenizer is None:
		logging.info(f"Load tokenizer from {model_name_or_path}")
		tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
	if dataset is None:
		logging.info(f"Load dataset from {dataset_name_or_path} with split {test_data_split}")
		dataset = load_dataset(dataset_name_or_path, split=test_data_split)
	if model_name_or_path is None:
		model_name_or_path = model.config.name_or_path
	logging.info(f"Evaluating {model_name_or_path} ...")
	if test_data_size is not None:
		if test_data_size < 1:
			test_data_size = int(len(dataset) * test_data_size)
			dataset = dataset.select(range(int(test_data_size)))
		else:
			dataset = dataset.select(range(min(test_data_size, len(dataset))))
	logging.info(f"Dataset ({len(dataset)} samples): {dataset.cache_files if dataset_name_or_path is None else dataset_name_or_path}")
	metric_summary = dict()
	for _, _, metric_name in metrics:
		if metric_name in metric_summary:
			logging.warning(f"Duplicate metrics: {metric_name}")
		metric_summary[metric_name] = {"history": list()}
	for i, data in enumerate(dataset):
		input_text = data[input_column]	# Str
		target_text = data[target_column]	# Str
		logging.info(f"Test data {i}: \n  - Input: {input_text}\n  - Target: {target_text}")
		input_token_ids = tokenizer.encode(input_text, return_tensors=None)	# List[<token_id>]
		target_token_ids = tokenizer.encode(target_text, return_tensors=None)	# List[<token_id>]
		if do_sample:
			for j in range(do_sample_times):
				logging.info(f"  - Sample {j}")
				predict_text, predict_token_prob, predict_logits = generate_token_prob(
					model = model,
					tokenizer = tokenizer,
					prompt = input_text,
					max_length = target_max_length,
					generate_kwargs = {"do_sample": True, "use_cache": use_cache, **do_sample_kwargs},
					device = device,
				)	# predict_token_prob: List[Tuple(Int, Str, Float)]
				logging.info(f"    - Predict text: {predict_text}")
				predict_token_ids = [predict_token_prob[i][0] for i in range(len(predict_token_prob))]	# List[<token_id>]
				sample_history = {metric_name: list() for metric_name in metric_summary}
				for metric_function_name, metric_kwargs, metric_name in metrics:
					if metric_function_name in ["calc_token_accuracy", "calc_bleu"]:
						# Return single value
						returned_value = eval(metric_function_name)(predict=predict_token_ids, target=target_token_ids, **metric_kwargs)
						sample_history[metric_name].append(returned_value)
					elif metric_function_name in ["calc_rouge_n", "calc_rouge_w"]:
						# Return Dict: Precision Revall F1-score
						returned_value = eval(metric_function_name)(predict=predict_token_ids, target=target_token_ids, **metric_kwargs)
						precision, recall, f1_score = returned_value["precision"], returned_value["recall"], returned_value["f1_score"]
						sample_history[metric_name].append((precision, recall, f1_score))
					elif metric_function_name in ["calc_perplexity"]:
						# Special metric calculation: positional arguments are not `predict` and `target`
						returned_value = eval(metric_function_name)(prompt=predict_token_ids, completion=target_token_ids, model=model, **metric_kwargs)
						sample_history[metric_name].append(returned_value)
					else:
						logging.warning(f"Unknown metric function name: {metric_function_name}")
						continue
					logging.info(f"    - {metric_name}: {returned_value}")
				for metric_name in sample_history:
					metric_summary[metric_name]["history"].append(sample_history[metric_name])
			for metric_name in metric_summary:
				if metric_function_name in ["calc_token_accuracy", "calc_bleu", "calc_perplexity"]:
					# Single value
					metric_summary[metric_name]["sample_mean"] = [numpy.mean(history) for history in metric_summary[metric_name]["history"]]
					metric_summary[metric_name]["sample_std"] = [numpy.std(history) for history in metric_summary[metric_name]["history"]]
					metric_summary[metric_name]["population_mean"] = numpy.mean(metric_summary[metric_name]["sample_mean"])
					metric_summary[metric_name]["population_std"] = numpy.std(metric_summary[metric_name]["population_std"])
				elif metric_function_name in ["calc_rouge_n", "calc_rouge_w"]:
					# Multiple values
					metric_summary[metric_name]["sample_mean"] = [[numpy.mean(history[i]) for i in range(len(history))] for history in metric_summary[metric_name]["history"]]
					metric_summary[metric_name]["sample_std"] = [[numpy.std(history[i]) for i in range(len(history))] for history in metric_summary[metric_name]["history"]]
					metric_summary[metric_name]["population_mean"] = numpy.mean(metric_summary[metric_name]["sample_mean"], axis=0).tolist()
					metric_summary[metric_name]["population_std"] = numpy.std(metric_summary[metric_name]["sample_std"], axis=0).tolist()
				else:
					logging.warning(f"Unknown metric function name: {metric_function_name}")
					continue
		else:
			predict_text, predict_token_prob, predict_logits = generate_token_prob(
				model = model,
				tokenizer = tokenizer,
				prompt = input_text,
				max_length = target_max_length,
				generate_kwargs = {"do_sample": False, "use_cache": use_cache, **do_sample_kwargs},
				device = device,
			)	# predict_token_prob: List[Tuple(Int, Str, Float)]
			logging.info(f"    - Predict text: {predict_text}")
			predict_token_ids = [predict_token_prob[i][0] for i in range(len(predict_token_prob))]	# List[<token_id>]
			for metric_function_name, metric_kwargs, metric_name in metrics:
				if metric_function_name in ["calc_token_accuracy", "calc_bleu"]:
					# Return single value
					returned_value = eval(metric_function_name)(predict=predict_token_ids, target=target_token_ids, **metric_kwargs)
					metric_summary[metric_name]["history"].append(returned_value)
				elif metric_function_name in ["calc_rouge_n", "calc_rouge_w"]:
					# Return Dict: Precision Revall F1-score
					returned_value = eval(metric_function_name)(predict=predict_token_ids, target=target_token_ids, **metric_kwargs)
					precision, recall, f1_score = returned_value["precision"], returned_value["recall"], returned_value["f1_score"]
					metric_summary[metric_name]["history"].append((precision, recall, f1_score))
				elif metric_function_name in ["calc_perplexity"]:
					# Special metric calculation: positional arguments are not `predict` and `target`
					returned_value = eval(metric_function_name)(
						prompt = input_token_ids, 
						completion = target_token_ids, 
						model = model, 
						device = device, 
						**metric_kwargs,
					)
					metric_summary[metric_name]["history"].append(returned_value)
				else:
					logging.warning(f"Unknown metric function name: {metric_function_name}")
					continue
				logging.info(f"    - {metric_name}: {returned_value}")
			for metric_name in metric_summary:
				if metric_function_name in ["calc_token_accuracy", "calc_bleu", "calc_perplexity"]:
					# Single value
					metric_summary[metric_name]["population_mean"] = numpy.mean(metric_summary[metric_name]["history"])
					metric_summary[metric_name]["population_std"] = numpy.mean(metric_summary[metric_name]["history"])
				elif metric_function_name in ["calc_rouge_n", "calc_rouge_w"]:
					# Multiple values
					metric_summary[metric_name]["population_mean"] = numpy.mean(metric_summary[metric_name]["history"], axis=0).tolist()
					metric_summary[metric_name]["population_std"] = numpy.mean(metric_summary[metric_name]["history"], axis=0).tolist()
				else:
					logging.warning(f"Unknown metric function name: {metric_function_name}")
					continue
	return metric_summary
""",
# ----
r"""
# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@stu.sufe.edu.cn

import re
import torch
import logging
from torch.nn import functional as F
from matplotlib import pyplot as plt
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

from src.tools.plot import plot_tensor_heatmap
from src.tools.hook import register_forward_hook_decorator
from src.tools.transformers import greedy_decode, robust_cosine_similarity, robust_corrcoef
from src.pipelines.generate import display_pipeline
from src.modules import (
	ParallelQwen2Model, SkipLayerQwen2ForCausalLM,
	ParallelQwen2ForCausalLM, SkipLayerQwen2ForCausalLM,
	ParallelQwen3Model, SkipLayerQwen3ForCausalLM,
	ParallelQwen3ForCausalLM, SkipLayerQwen3ForCausalLM,
	ParallelLlamaModel, SkipLayerLlamaModel,
	ParallelLlamaForCausalLM, SkipLayerLlamaForCausalLM,
)

# Horizontal comparison: Compare hook data (which comes from different prompts) by module names
# Focusing on comparing the inputs or outputs of the same modules in different hooks
# @param hook_datas: List[Dict] of length 2, i.e. currently we only compare two series of hook data
# @param hook_data_paths: List[Str], default None but at least one of `hook_datas` and `hook_data_paths` is not None
# @param hook_module_names: List[Str], e.g. ["model.layers[0].self_attn.q_proj", "model.layers[0].self_attn.k_proj", "model.layers[0].self_attn.v_proj"]
# @param hook_module_name_suffixes: List[Str], e.g. ["q_proj", "k_proj", "v_proj"]
# @param comparison_index: List[Str], e.g. ["mean_diff", "max_diff", "corr"]
# @param max_length: [Int] when generating one token, one comparison is conducted. So we need to limit the max comparison by `max_length`
# @param figure_size: [Int] default 5
# @param outlier_ratio: [Float] Default 0 means not filtering outlier. Setting as a 0-1 ratio to filter outliers
def horizontal_comparison_of_forward_hook(
	hook_datas = None,
	hook_data_paths = None,
	hook_module_names = ["model.layers[0].self_attn.q_proj", "model.layers[1].self_attn.k_proj", "model.layers[2].self_attn.v_proj",],
	hook_module_name_suffixes = ["q_proj", "k_proj", "v_proj"],
	comparison_index = ["mean_diff", "max_diff", "corr", "sim", "robust_corr", "robust_sim"],
	max_length = 999,
	figure_size = 5,
	outlier_ratio = 0.,
):
	regex = re.compile("\[\d+\]", re.I)	# e.g. Used to match `[0]` in `model.layers[0].self_attn.q_proj`
	assert hook_datas is not None or hook_data_paths is not None
	if hook_datas is None:
		hook_datas = [torch.load(hook_data_path) for hook_data_path in hook_data_paths]
	for token_i, hook_data_group in enumerate(zip(*hook_datas)):
		if token_i >= max_length:
			break
		# Hook data when generating token i
		# Summary dictionary contains all the compared index
		comparison_summary_dict = {
			index: {
				module_name_suffix: {"input": list(), "output": list()}
				for module_name_suffix in hook_module_name_suffixes
			} for index in ["mean_diff", "max_diff", "corr", "sim", "robust_corr", "robust_sim"]
		}
		for module_name in hook_module_names:
			module_name_suffix = module_name.split('.')[-1]
			module_name_suffix = regex.sub(str(), module_name_suffix)
			if module_name_suffix in hook_module_name_suffixes:
				# 1. Process inputs in hook data
				input_data_group = [data[module_name].get("input", data[module_name].get("args")) for data in hook_data_group]
				for j, input_data in enumerate(input_data_group):
					# Assertation for ensuring data format of inputs
					assert len(input_data) == 1 and isinstance(input_data[0], tuple)
					if len(input_data[0]) > 1:
						logging.warning(f"Input data {j} has more than 1 components: {len(input_data[0])}")
				input_tensors = [input_data[0][0].float() for input_data in input_data_group]
				for j, input_tensor in enumerate(input_tensors):
					logging.info(f"Size of input tensor {j}: {input_tensor.size()}")
				# 2. Process outputs in hook data
				output_data_group = [data[module_name]["output"] for data in hook_data_group]
				output_tensors = list()
				for j, output_data in enumerate(output_data_group):
					# Assertation for ensuring data format of outputs
					assert len(output_data) == 1
					if isinstance(output_data[0], torch.Tensor):
						output_tensor = output_data[0]
					else:
						assert isinstance(output_data[0], tuple)
						if len(output_data[0]) > 1:
							logging.warning(f"Output data {j} has more than 1 components: {len(output_data[0])}")
						output_tensor = output_tensor[0][0]
					output_tensors.append(output_tensor)

				# 3. Summary data calculation
				## 3.1 Calculate Mean Difference
				input_diff = input_tensors[0] - input_tensors[1]
				output_diff = output_tensors[0] - output_tensors[1]
				mean_input_diff = torch.norm(input_diff, p="fro") / input_tensors[0].numel()
				mean_output_diff = torch.norm(output_diff, p="fro") / output_tensors[0].numel()
				comparison_summary_dict["mean_diff"][module_name_suffix]["input"].append(mean_input_diff.item())
				comparison_summary_dict["mean_diff"][module_name_suffix]["output"].append(mean_output_diff.item())
				## 3.2 Calculate Max Difference
				max_input_diff = torch.max(torch.abs(input_diff)).item()
				max_output_diff = torch.max(torch.abs(output_diff)).item()
				comparison_summary_dict["max_diff"][module_name_suffix]["input"].append(max_input_diff)
				comparison_summary_dict["max_diff"][module_name_suffix]["output"].append(max_output_diff)
				## 3.3 Calculate Correlation Coefficient
				input_corr = torch.corrcoef(torch.stack([input_tensors[0].flatten(), input_tensors[1].flatten()]))[0, 1].item()
				output_corr = torch.corrcoef(torch.stack([output_tensors[0].flatten(), output_tensors[1].flatten()]))[0, 1].item()
				comparison_summary_dict["corr"][module_name_suffix]["input"].append(input_corr)
				comparison_summary_dict["corr"][module_name_suffix]["output"].append(output_corr)
				## 3.4 Calculate Similarity
				input_sim = F.cosine_similarity(input_tensors[0].flatten(), input_tensors[1].flatten(), dim=0).item()
				output_sim = F.cosine_similarity(output_tensors[0].flatten(), output_tensors[1].flatten(), dim=0).item()
				comparison_summary_dict["sim"][module_name_suffix]["input"].append(input_sim)
				comparison_summary_dict["sim"][module_name_suffix]["output"].append(output_sim)
				## 3.5 Robust Correlation Coefficient
				input_robust_corr = robust_corrcoef(input_tensors[0], input_tensors[1], outlier_ratio = outlier_ratio)
				output_robust_corr = robust_corrcoef(output_tensors[0], output_tensors[1], outlier_ratio = outlier_ratio)
				comparison_summary_dict["robust_corr"][module_name_suffix]["input"].append(input_robust_corr)
				comparison_summary_dict["robust_corr"][module_name_suffix]["output"].append(output_robust_corr)
				## 3.6 Robust Similarity
				input_robust_sim = robust_cosine_similarity(input_tensors[0], input_tensors[1], outlier_ratio = outlier_ratio)
				output_robust_sim = robust_cosine_similarity(output_tensors[0], output_tensors[1], outlier_ratio = outlier_ratio)
				comparison_summary_dict["robust_sim"][module_name_suffix]["input"].append(input_robust_sim)
				comparison_summary_dict["robust_sim"][module_name_suffix]["output"].append(output_robust_sim)
				# ...
		nrows, ncols = len(comparison_index), len(hook_module_name_suffixes)
		fig, axes = plt.subplots(
			nrows = nrows,
			ncols = ncols,
			figsize = (figure_size * 1.2 * ncols, figure_size * nrows),
		)
		for i, summary_key in enumerate(comparison_index):
			for j, module_name_suffix in enumerate(hook_module_name_suffixes):
				y_input = comparison_summary_dict[summary_key][module_name_suffix]["input"]
				y_output = comparison_summary_dict[summary_key][module_name_suffix]["output"]
				assert len(y_input) == len(y_output)
				x = range(len(y_input))
				if len(x) == 0:
					# No inputs exist
					continue
				if len(comparison_index) == 1 and len(hook_module_name_suffixes) == 1:
					target_ax = axes
				elif len(comparison_index) == 1:
					target_ax = axes[j]
				elif len(hook_module_name_suffixes) == 1:
					target_ax = axes[i]
				else:
					target_ax = axes[i, j]
				target_ax.bar(x, y_input, label=f"input_{summary_key}", alpha=.5)
				target_ax.bar(x, y_output, label=f"output_{summary_key}", alpha=.5)
				target_ax.set_xlabel("Layer #"), target_ax.set_ylabel(summary_key), target_ax.set_title(f"{summary_key} for {module_name_suffix} on token {token_i}")
				target_ax.legend()
		plt.show(), plt.close()

# Vertical comparison: Compare data in a single hook
# Focusing on comparing the inputs and outputs of the same modules
# @param hook_data: [Dict] Hook data object
# @param hook_data_path: [Str] Default None but at least one of `hook_datas` and `hook_data_paths` is not None
# @param hook_module_names: List[Str], e.g. ["model.layers[0]"]
# @param comparison_index: List[Str], e.g. ["mean_diff", "max_diff", "corr"]
# @param max_length: [Int] when generating one token, one comparison is conducted. So we need to limit the max comparison by `max_length`
# @param figure_size: [Int] Default 5
# @param watched_module_names: List[Int], you can selected several module here to plot heat map of input-output difference
# @param outlier_ratio: [Float] Default 0 means not filtering outlier. Setting as a 0-1 ratio to filter outliers
def vertical_comparison_of_forward_hook(
	hook_data = None,
	hook_data_path = None,
	hook_module_names = ["model.layers[0]", "model.layers[1]", "model.layers[2]"],
	comparison_index = ["mean_diff", "max_diff", "corr", "sim", "robust_corr", "robust_sim"],
	max_length = 999,
	figure_size = 5,
	watched_module_names = ["model.layers[0]"],
	outlier_ratio = 0.,
):
	assert hook_data is not None or hook_data_path is not None
	if hook_data is None:
		hook_data = torch.load(hook_data_path, weights_only=False)
	for token_i in range(max_length):
		comparison_summary_dict = {index: list() for index in ["mean_diff", "max_diff", "corr", "sim", "robust_corr", "robust_sim"]}
		# Plot heatmap of input-output difference
		if watched_module_names:
			fig, axes = plt.subplots(1, len(watched_module_names), figsize=(1.2 * 5 * figure_size * len(watched_module_names), figure_size))
		subplot_index = -1
		for module_name in hook_module_names:
			input_tensor = hook_data[token_i][module_name].get("input", hook_data[token_i][module_name].get("args"))[0][0]
			output_tensor = hook_data[token_i][module_name]["output"][0][0]
			diff = input_tensor - output_tensor
			mean_diff = torch.norm(diff, p="fro") / input_tensor.numel()
			max_diff = torch.max(torch.abs(diff)).item()
			corr = torch.corrcoef(torch.stack([input_tensor.flatten(), output_tensor.flatten()]))[0, 1].item()
			sim = F.cosine_similarity(input_tensor.flatten(), output_tensor.flatten(), dim=0).item()
			robust_corr = robust_corrcoef(input_tensor, output_tensor, outlier_ratio = outlier_ratio)
			robust_sim = robust_cosine_similarity(input_tensor, output_tensor, outlier_ratio = outlier_ratio)
			comparison_summary_dict["mean_diff"].append(mean_diff.item())
			comparison_summary_dict["max_diff"].append(max_diff)
			comparison_summary_dict["corr"].append(corr)
			comparison_summary_dict["sim"].append(sim)
			comparison_summary_dict["robust_corr"].append(robust_corr)
			comparison_summary_dict["robust_sim"].append(robust_sim)
			if module_name in watched_module_names:
				subplot_index += 1
				assert diff.size(0) == 1
				plot_tensor_heatmap(
					tensor = torch.abs(diff)[0, :, :],
					ax = axes[subplot_index] if len(watched_module_names) > 1 else axes,
					is_show=False,
					title=f"Diff in {module_name} of Token {token_i}",
				)
		if watched_module_names:
			plt.show(), plt.close()
		# Plot line chart of comparison index
		ncols = len(comparison_index)
		nrows = 1
		fig, axes = plt.subplots(
			nrows = nrows,
			ncols = ncols,
			figsize = (ncols * figure_size * 1.2, nrows * figure_size),
		)
		for c, summary_key in enumerate(comparison_index):
			for r in range(nrows):
				if ncols == 1 and nrows == 1:
					target_ax = axes
				elif ncols == 1:
					target_ax = axes[r]
				elif nrows == 1:
					target_ax = axes[c]
				else:
					target_ax = axes[r, c]
				x = list(range(len(hook_module_names)))
				target_ax.plot(x, comparison_summary_dict[summary_key], label=summary_key, marker='o')
				# Plot text on each dot
				if summary_key in ["corr", "sim", "robust_corr", "robust_sim"]:
					for i, (x_i, y_i) in enumerate(zip(x, comparison_summary_dict[summary_key])):
						text_flag = i % 3 == 0
						last_y_flag = abs(y_i - comparison_summary_dict[summary_key][i - 1]) > .1 if i > 0 else True
						next_y_flag = abs(y_i - comparison_summary_dict[summary_key][i + 1]) > .1 if i < len(x) - 1 else True
						if text_flag or (last_y_flag and next_y_flag):
							target_ax.text(x_i, y_i, str(round(y_i, 3)), ha="center", va="bottom", fontsize=12, color="red")

				target_ax.legend(), target_ax.set_xlabel("Layer #"), target_ax.set_ylabel(summary_key), target_ax.set_title(f"{comparison_index[c]} on token {token_i}")
		plt.show(), plt.close()

# Generating by skipping decoder blocks
# Focusing on the generating results under skipping different layers
# *** VERY IMPORT NOTICE ***
# Notice that no matter what `skip_layer_ids` is, the `forward_hook_module_names` (as well as `backward_hook_module_names`) should be always range from 0-`num_hidden_layers-1`
# However, to those layerId in `skip_layer_ids`, the hook data is None
# This is different from `easy_skip_layer_generation`
# @param Model: AutoModel class, e.g. Qwen2Model
# @param ModelForCausalLM: AutoModelForCausalLM class, e.g. Qwen2ForCausalLM
# @param model_name_or_path: Str
# @param tokenizer: HuggingFace tokenizer object
# @param prompt: Str
# @param max_length: Int
# @param device: [Str] e.g. "cuda" or "cpu"
# @param skip_layer_ids: List[Int], Layer # to be skipped
# @param use_kv_cache: [Boolean]
# @param forward_hook_module_names: [List[Str]] Default None, otherwise register backward hook for `forward_hook_module_names`, e.g. ["model.layers[0].self_attn.q_proj", "model.layers[0].self_attn.k_proj"]
# @param backward_hook_module_names: [List[Str]] Default None, otherwise register backward hook for `backward_hook_module_names`, e.g. ["model.layers[0].self_attn.q_proj", "model.layers[0].self_attn.k_proj"]
def skip_layer_generation(
	SkipLayerModelForCausalLM,
	model_name_or_path,
	tokenizer,
	prompt,
	max_length,
	device,
	skip_layer_ids = list(),
	use_kv_cache = True,
	forward_hook_module_names = None,
	backward_hook_module_names = None,
):
	config = AutoConfig.from_pretrained(model_name_or_path)
	model = SkipLayerModelForCausalLM.from_pretrained(
		model_name_or_path,
		config = config,
		skip_layer_ids = skip_layer_ids,
	).to(device)
	results = greedy_decode(
		model,
		tokenizer,
		prompt = prompt,
		max_length = max_length,
		device = device,
		use_kv_cache = use_kv_cache,
		forward_hook_module_names = forward_hook_module_names,
		backward_hook_module_names = backward_hook_module_names,
	)
	return results

# Generating by skipping decoder blocks (Need not load model each time)
# Focusing on the generating results under skipping different layers
# *** VERY IMPORT NOTICE ***
# Notice that if `skip_layer_ids = [0]`, then `forward_hook_module_names` (as well as `backward_hook_module_names`) should be like ["model.layers[1]", "model.layers[2]", ...]
# None of the hook data should be `None`
# @param model: HuggingFace AutoModelForCausalLM object
# @param tokenizer: HuggingFace tokenizer object
# @param prompt: Str
# @param max_length: Int
# @param device: [Str] e.g. "cuda" or "cpu"
# @param skip_layer_ids: List[Int], Layer # to be skipped
# @param forward_hook_module_names: [List[Str]] Default None, otherwise register backward hook for `forward_hook_module_names`, e.g. ["model.layers[0].self_attn.q_proj", "model.layers[0].self_attn.k_proj"]
# @param backward_hook_module_names: [List[Str]] Default None, otherwise register backward hook for `backward_hook_module_names`, e.g. ["model.layers[0].self_attn.q_proj", "model.layers[0].self_attn.k_proj"]
def easy_skip_layer_generation(
	model,
	tokenizer,
	prompt,
	max_length,
	device,
	skip_layer_ids = list(),
	use_kv_cache = True,
	forward_hook_module_names = None,
	backward_hook_module_names = None,
):
	if skip_layer_ids:
		# 1. Delete `self.layers` and modify `layer.self_attn.layer_idx`
		filtered_layers = list()
		backup_layer_ids = list()
		backup_layers = model.model.layers
		for layer_id, layer in enumerate(model.model.layers):
			if layer_id not in skip_layer_ids:
				backup_layer_ids.append(layer_id)
				layer.self_attn.layer_idx = len(filtered_layers)
				filtered_layers.append(layer)
		model.model.layers = torch.nn.ModuleList(filtered_layers)
		# 2. Delete `self.config.layer_types`
		# back_up_layer_types = model.model.config.layer_types[:]
		# filtered_layer_types = [
			# layer_type for layer_id, layer_type in enumerate(model.model.config.layer_types)
			# if layer_id not in skip_layer_ids
		# ]
		# 3. Minus `self.config.num_hidden_layers`
		model.model.config.num_hidden_layers -= len(skip_layer_ids)
	results = greedy_decode(
		model,
		tokenizer,
		prompt = prompt,
		max_length = max_length,
		device = device,
		use_kv_cache = use_kv_cache,
		forward_hook_module_names = forward_hook_module_names,
		backward_hook_module_names = backward_hook_module_names,
	)
	# Recover for follow-up callback
	if skip_layer_ids:
		# 1. Recover `self.layers`
		assert len(backup_layer_ids) == len(filtered_layers)
		for back_up_layer_id, layer in zip(backup_layer_ids, filtered_layers):
			layer.self_attn.layer_idx = back_up_layer_id
		model.model.layers = backup_layers
		# 2. Recover `self.config.layer_types`
		# model.model.config.layer_types = back_up_layer_types[:]
		# 3. Recover `self.config.num_hidden_layers`
		model.model.config.num_hidden_layers += len(skip_layer_ids)
	return results


# Generating by edit layer parameters or layer inputs
# @param edit_layer_id: Int
# @param edit_layer_input: torch.Tensor
def easy_edit_layer_generation(
	model,
	edit_layer_id,
	edit_layer_input,
):
	NotImplemented
""",
# ----
r"""
# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@stu.sufe.edu.cn

import torch
import pandas
import logging
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.tools.transformers import greedy_decode, k_step_greedy_decode, beam_search_decode, generate_token_prob, get_generation_eos_token_ids
from src.tools.hook import register_forward_hook_decorator, register_backward_hook_decorator
from src.modules import (
	ParallelQwen2Model, SkipLayerQwen2ForCausalLM, 
	ParallelQwen2ForCausalLM, SkipLayerQwen2ForCausalLM, 
	ParallelQwen3Model, SkipLayerQwen3ForCausalLM, 
	ParallelQwen3ForCausalLM, SkipLayerQwen3ForCausalLM, 
	ParallelLlamaModel, SkipLayerLlamaForCausalLM, 
	ParallelLlamaForCausalLM, SkipLayerLlamaForCausalLM, 
)

# Do only one time forward to check model outputs
# @param model: Huggingface AutoModel object
def one_time_forward_pipeline(
	model, 
	tokenizer,
	prompt,
	device,
	forward_hook_module_names = None,
	backward_hook_module_names = None,
):
	if backward_hook_module_names is not None:
		raise NotImplementedError("Currently not support `backward_hook_module_names`")
	@register_forward_hook_decorator(module_names = forward_hook_module_names)
	def easy_forward(inputs, *, model, **kwargs):
		return model(inputs, **kwargs)
	if device is None:
		device = "cuda" if torch.cuda.is_available() else "cpu"
	logging.info(f"Device: {device}")	
	inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)	# Str => Long(1, n_tokens)
	outputs = easy_forward(inputs, model=model, use_cache=False)
	hook_data = outputs.hook_outputs
	return hook_data


# Display generation details token by token
# @param tokenizer: Huggingface tokenizer Object
# @param text: [Str] Final generated text
# @param token_prob: List[Tuple(Int, Str, Float)], `len(generated_id_prob)` is `max_length`, indicating the generated probability of each token
# @param logits: Tuple[FloatTensor(1, n_vocab)], `len(generated_logits)` is `max_length`, indicating the logits when each token is generated
# @param k: [Int] top-k decode candidates to be display
# @param eos_token_id: [Int] tokenId of <eos> token, e.g. 151643(<|endoftext|>) for Qwen model
# @return df_display: [pandas.DataFrame] ["id", "token", "prob", "max_id", "cand_tokens", "cand_probs", "eos_prob"]
def display_pipeline(tokenizer,
					 text,
					 token_probs,
					 logits,
					 k = 3,
					 eos_token_id = 151643,
					 ):
	df_token_probs = pandas.DataFrame(token_probs, columns=["id", "token", "prob"])
	def _display_tensor(_tensor, _round):
		return list(map(lambda x: round(x, _round), _tensor.tolist()))
	df_display = {
		"max_id": [],
		"cand_tokens": [],
		"cand_probs": [],
		"eos_prob": [],
	}
	for tensor in logits:
		tensor_to_prob = F.softmax(tensor[0], dim=-1)
		top_k = torch.topk(tensor_to_prob, k = 3)
		top_k_values = top_k.values
		top_k_indices = top_k.indices
		max_id = top_k_indices[0].item()
		probs = _display_tensor(top_k_values, 4)
		cand_ids = _display_tensor(top_k_indices, 4)
		cand_tokens = [tokenizer.decode(token_id) for token_id in top_k_indices]
		eos_prob = tensor_to_prob[eos_token_id].item()
		df_display["max_id"].append(max_id)
		df_display["cand_tokens"].append(cand_tokens)
		df_display["cand_probs"].append(probs)
		df_display["eos_prob"].append(eos_prob)
	df_display = pandas.DataFrame(df_display, columns=["max_id", "cand_tokens", "cand_probs", "eos_prob"])
	return pandas.concat([df_token_probs, df_display], axis=1)

# Generate tokens by a given prompt, using `model.generate`
# @param model_name_or_path: [Str]
# @param prompt: [Str]
# @param max_length: [Int]
# @param device: [Str|torch.device] e.g. "cuda", "cpu", torch.device("cpu")
# @param generate_kwargs: [Dict] keyword arguments for `model.generate`
# @return df_display: the returned of `display_pipeline`
def generate_pipeline(model_name_or_path,
					  prompt,
					  max_length,
					  device = None,
					  generate_kwargs = None,
					  model_parallel_class = None,
					  n_cuda = 2,
					  ):
	logging.info("Load model and tokenizer ...")
	if device is None:
		device = "cuda" if torch.cuda.is_available() else "cpu"
	logging.info(f"Device: {device}")
	tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
	if model_parallel_class is None:
		model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True).to(device)
	else:
		model = eval(model_parallel_class).from_pretrained(model_name_or_path, n_cuda=n_cuda)
		model.module_to_device()
	eos_token_ids = get_generation_eos_token_ids(model)
	logging.info(f"  - EOS Tokens: {eos_token_ids}")
	logging.info("Model Generate ...")
	if generate_kwargs is None:
		# Greedy decode configurations
		generate_kwargs = {"do_sample": False, "top_k": 0, "top_p": 1., "num_beams": 1, "temperature": 1}
	text, token_prob, logits = generate_token_prob(model, tokenizer, prompt, max_length, generate_kwargs, device)
	logging.info(f"Generated text: {text}")
	return display_pipeline(tokenizer, text, token_prob, logits, eos_token_id=eos_token_ids[0])
""",
# ----
r"""
# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@stu.sufe.edu.cn

import trl
import torch
import logging
from copy import deepcopy
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, HfArgumentParser
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import (
	ScriptArguments, ModelConfig, 
	# SFTConfig, SFTTrainer,
	# PPOConfig, PPOTrainer,
	# DPOConfig, DPOTrainer,
	# GRPOConfig, GRPOTrainer,
	get_peft_config, get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
from src.tools.trl import update_trl_config, generate_simple_data_processor
from src.modules import (
	ParallelQwen2Model, SkipLayerQwen2ForCausalLM, 
	ParallelQwen2ForCausalLM, SkipLayerQwen2ForCausalLM, 
	ParallelQwen3Model, SkipLayerQwen3ForCausalLM, 
	ParallelQwen3ForCausalLM, SkipLayerQwen3ForCausalLM, 
	ParallelLlamaModel, SkipLayerLlamaForCausalLM, 
	ParallelLlamaForCausalLM, SkipLayerLlamaForCausalLM, 
)

# Trainer Pipeline
# @param name: [Str] e.g. "SFT", "PPO", "DPO", "GRPO"
# @param data_processor: Function object prepared for `dataset.map(data_processor)`
# @param trainer_config: [Dict, peft.XXXConfig] including keyword arguments, e.g. 
# @param model_config: [Dict, peft.ModelConfig] including keyword arguments, e.g. 
# @param script_arguments: [Dict, peft.ScriptArguments] including keyword arguments, e.g. "dataset_name", "dataset_train_split", "dataset_test_split"
# @param config_kwargs: [Dict] keyword arguments for updating TRL-Config, `ScriptArguments`, `ModelConfig`
#   - keyword arguments for `TRLConfig`: e.g. "output_dir", "adam_xxx", "learning_rate", "kl_coef", "push_to_hub"
#   - keyword arguments for `ScriptArguments`: e.g. "output_dir", "adam_xxx", "learning_rate", "kl_coef", "push_to_hub"
#   - keyword arguments for `ModelConfig`: e.g. "model_name_or_path", "torch_dtype", "trust_remote_code", "use_peft", "lora_xxx", "load_in_4bit", "bnb_4bit_compute_dtype", "bnb_4bit_quant_type"
# @param trainer_kwargs: [Dict] keyword arguments for updating TRL-Trainer
#   - keyword arguments for all Trainers: e.g. "data_collator", "callbacks"
#   - keyword arguments for `SFTTrainer`: e.g. "compute_loss_func", "compute_metrics"
#   - keyword arguments for `PPOTrainer`: e.g. "ref_model[required]", "reward_model[required]", "value_model[required]"
#   - keyword arguments for `DPOTrainer`: e.g. "ref_model"
#   - keyword arguments for `GRPOTrainer`: e.g. "reward_funcs[required]"
#@param parallel_model_class: [Str] e.g. "ParallelQwen2ForCausalLM", "ParallelQwen2Model", default `None` refer to AutoModelForCausalLM
def base_pipeline(name, data_processor, config_kwargs, trainer_kwargs, parallel_model_class = None, n_cuda = 2):
	# 1 Configuration
	TRLConfig, TRLTrainer = getattr(trl, f"{name}Config"), getattr(trl, f"{name}Trainer")
	parser = HfArgumentParser((ScriptArguments, TRLConfig, ModelConfig))
	script_arguments, trainer_config, model_config = parser.parse_args_into_dataclasses()
	script_arguments = update_trl_config(script_arguments, **config_kwargs)
	trainer_config = update_trl_config(trainer_config, **config_kwargs)
	model_config = update_trl_config(model_config, **config_kwargs)
	peft_config = get_peft_config(model_config)
	quantization_config = get_quantization_config(model_config)
	# 2 Load models and tokenizer
	logging.info("Load models and tokenizer ...")
	logging.info(f"  - Model: {model_config.model_name_or_path}")
	tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
	if tokenizer.chat_template is None:
		tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
	if parallel_model_class is None:
		logging.info("Using AutoModelForCausalLM ...")
		model = AutoModelForCausalLM.from_pretrained(
			model_config.model_name_or_path,
			device_map = "auto",
			trust_remote_code = model_config.trust_remote_code,
			quantization_config = quantization_config,
		)
	else:
		logging.info(f"Using {parallel_model_class} ...")
		model = eval(parallel_model_class).from_pretrained(
			model_config.model_name_or_path,
			n_cuda = n_cuda,
			device_map = "cpu",
		)
	if peft_config is not None:
		logging.info("Prepare model for PEFT ...")
		model.config.pretraining_tp = 1
		model.config.use_cache = False
		model.gradient_checkpointing_enable()
		# If `prepare_model_for_kbit_training` is ignored, and `gradient_checkpointing = True` (for GPU memory saving)
		# Then you need set `model.enable_input_require_grads()` yourself
		# model = prepare_model_for_kbit_training(model)
		model.enable_input_require_grads()
		model = get_peft_model(model, peft_config)

	if name == "PPO":
		logging.info("PPO load reward value and reference models ...")
		# PPO is special! It needs more components!
		logging.info(f"  - Reward model: {trainer_config.reward_model_path}")
		reward_model = AutoModelForSequenceClassification.from_pretrained(
			trainer_config.reward_model_path,
			trust_remote_code = model_config.trust_remote_code,
			num_labels = 1,
		)
		value_model = AutoModelForSequenceClassification.from_pretrained(
			trainer_config.reward_model_path,
			trust_remote_code = model_config.trust_remote_code,
			num_labels = 1,
		)
		reward_tokenizer = AutoTokenizer.from_pretrained(trainer_config.reward_model_path)
		if reward_tokenizer.chat_template is None:
			reward_tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
		logging.info("  - Copy reference model ...")
		
		# Clone model: I prefer deepcopy
		ref_model = deepcopy(model)
		# ref_model = model.__class__(model.config)
		# ref_model.load_state_dict(model.state_dict())
		
		trainer_kwargs["reward_model"] = reward_model
		trainer_kwargs["value_model"] = value_model
		trainer_kwargs["ref_model"] = ref_model
		trainer_kwargs["processing_class"] = reward_tokenizer
		logging.info("  - Done!")
		if data_processor is None:
			# The data processor of PPO is also different to others
			def data_processor(_data):
				outputs = tokenizer(_data["prompt"] + _data["completion"], padding = False)
				return {"input_ids": outputs["input_ids"]}
	else:
		trainer_kwargs["processing_class"] = tokenizer

	# 2 Load dataset
	logging.info("Load dataset ...")
	logging.info(f"  - Dataset: {script_arguments.dataset_name}")
	if data_processor is None:
		data_processor = generate_simple_data_processor(name)
	train_dataset = load_dataset(script_arguments.dataset_name, split=script_arguments.dataset_train_split)
	eval_dataset = load_dataset(script_arguments.dataset_name, split=script_arguments.dataset_test_split)
	train_dataset = train_dataset.map(data_processor, remove_columns=train_dataset.column_names)
	eval_dataset = eval_dataset.map(data_processor, remove_columns=eval_dataset.column_names)
	logging.info(f"  - Train dataset: {len(train_dataset)}")
	logging.info(f"  - Eval dataset: {len(eval_dataset)}")
	# 4 Train model
	logging.info("Trainer starts ...")
	trainer = TRLTrainer(
		model = model,
		args = trainer_config,
		train_dataset = train_dataset,
		eval_dataset = eval_dataset,
		peft_config = peft_config,
		**trainer_kwargs
	)
	trainer.train()
	logging.info("  - Trainer finishes!")
	# 5 Save model
	if trainer_config.push_to_hub:
		logging.info(f"  - Push checkpoints to {trainer_config.organization}/{trainer_config.push_to_hub_model_id}")
		trainer.push_to_hub()
	logging.info(f"Save model to {trainer_config.output_dir}")
	trainer.save_model(trainer_config.output_dir)

# SFT Pipeline
def sft_pipeline(data_processor, config_kwargs, trainer_kwargs, parallel_model_class = None, n_cuda = 2):
	base_pipeline(
		name = "SFT",
		data_processor = data_processor,
		config_kwargs = config_kwargs,
		trainer_kwargs = trainer_kwargs,
		parallel_model_class = parallel_model_class,
		n_cuda = n_cuda,
	)

# PPO Pipeline
def ppo_pipeline(data_processor, config_kwargs, trainer_kwargs, parallel_model_class = None, n_cuda = 2):
	base_pipeline(
		name = "PPO",
		data_processor = data_processor,
		config_kwargs = config_kwargs,
		trainer_kwargs = trainer_kwargs,
		parallel_model_class = parallel_model_class,
		n_cuda = n_cuda,
	)

# DPO Pipeline
def dpo_pipeline(data_processor, config_kwargs, trainer_kwargs, parallel_model_class = None, n_cuda = 2):
	base_pipeline(
		name = "DPO",
		data_processor = data_processor,
		config_kwargs = config_kwargs,
		trainer_kwargs = trainer_kwargs,
		parallel_model_class = parallel_model_class,
		n_cuda = n_cuda,
	)

# GRPO Pipeline
def grpo_pipeline(data_processor, config_kwargs, trainer_kwargs, parallel_model_class = None, n_cuda = 2):
	base_pipeline(
		name = "GRPO",
		data_processor = data_processor,
		config_kwargs = config_kwargs,
		trainer_kwargs = trainer_kwargs,
		parallel_model_class = parallel_model_class,
		n_cuda = n_cuda,
	)
""",
# ----
r"""下面介绍笔者对 **问题$4$** 结果（式$(7)$）的证明思路，这将涉及基础的概率论与微分方程的知识：

$\text{Proof}$：

定义在第一次采样的随机数为$x$的条件下（$0\le x\le 1$），期望上的猜测次数为$\mathbb E(x)$，那么需要计算的期望次数就是：
$$
T = \int_0^1\mathbb E(x)\text{d}x\tag{8}
$$
考察$x<1/2$与$x>1/2$两种情况：
$$
\mathbb E(x)=\left\{\begin{aligned}
&x\cdot 2+(1-x)\cdot\left[1+\frac{\int_{x}^1\mathbb E(t)\text{d}t}{1-x}\right]=1+x+\int_x^1\mathbb E(t)\text{d}t&&\text{if }x<\frac12\\
&(1-x)\cdot 2+x\cdot\left[1+\frac{\int_{0}^x\mathbb E(t)\text{d}t}{x}\right]=2-x+\int_{0}^x\mathbb E(t)\text{d}t&&\text{if }x>\frac12
\end{aligned}\right.\tag{9}
$$
以式$(9)$中$x<1/2$的情况为例（$x>1/2$的情况类似）：

- 有$x$的概率下一次采样比$x$还要小，此时就会猜测错误，因此猜测次数为$2$；
- 有$1-x$的概率下一次采样比$x$要大，此时猜测会正确，期望猜测次数就是中括号内的形式（<font color=red>思考</font>），这本质是一个条件期望形式；

由于式$(9)$中的$\mathbb E(x)$由两段构成，我们用不同的函数标记它：
$$
\mathbb E(x)=\left\{\begin{aligned}
&f(x)&&\text{if }x<\frac12\\
&g(x)&&\text{if }x>\frac12
\end{aligned}\right.\tag{10}
$$
其实很容易能够想得到，<font color=red>$\mathbb E(x)$是关于$x=1/2$对称的（因为$x$很大与$x$很小本质上没有区别，只要我第一次猜小或猜大就能抵消掉了）。</font>因此必然有：
$$
f(x)=g(1-x)\quad 0\le x<\frac12\tag{11}
$$
进一步地根据$\mathbb E(x)$的对称性，可得积分面积相同，即有：
$$
\left.\begin{aligned}
S_1&=\int_0^{1/2}\mathbb E(x)\text{d}x=\int_0^{1/2}f(x)\text{d}x\\
S_2&=\int_{1/2}^1\mathbb E(x)\text{d}x=\int_{1/2}^1g(x)\text{d}x
\end{aligned}\right\}\Rightarrow S_1=S_2=S\tag{12}
$$

根据下图可以更好的理解上面的对称性：

![在这里插入图片描述](https://i-blog.csdnimg.cn/blog_migrate/b08a2a7c45aa130f51d796c35e62b139.png#pic_center)


于是我们可以把式$(9)$中的积分改写成分段形式：
$$
\left\{\begin{aligned}
f(x)=&1+x+\int_{1/2}^1g(t)\text{d}t+\int_{x}^{1/2}f(t)\text{d}t&&\text{if }x<\frac12\\
g(x)=&2-x+\int_{0}^{1/2}f(t)\text{d}t+\int_{1/2}^xg(t)\text{d}t&&\text{if }x>\frac12
\end{aligned}\right.\tag{13}
$$
因为$f(x)$和$g(x)$其实是可以互相转换的，因此我们只需要式$(13)$的某一行即可求解结果，不妨使用第一行，即：
$$
f(x)=1+x+\int_{1/2}^1g(t)\text{d}t+\int_{x}^{1/2}f(t)\text{d}t=1+x+S_2+\int_{x}^{1/2}f(t)\text{d}x\tag{14}
$$
假设$f(x)$的原函数为$F(x)$，则可得：
$$
S_2=S_1=\int_{0}^{1/2}f(x)\text{d}x=F(1/2)-F(0)\quad \int_{x}^{1/2}f(t)\text{d}x=F(1/2)-F(x)\tag{15}
$$
将式$(15)$代入到式$(14)$中可得：
$$
f(x)=1+x+F(1/2)-F(0)+F(1/2)-F(x)=[1+2F(1/2)-F(0)]+x-F(x)\tag{16}
$$
式$(16)$中中括号内的内容都是常数，于是这就是一个常微分方程，设$y=F(x)$，则$y'=f(x)$，我们有

$$
y'=[1+2F(1/2)-F(0)]+x-y\tag{17}
$$

对式$(17)$两边求两阶导，可以转换为齐次的常微分方程：
$$
y'''+y''=0\tag{18}
$$
易得特征方程$r^2+r=0$的根为$r_1=0$和$r_2=-1$，因此式$(18)$的通解为：
$$
y'=C_1+C_2e^{-x}\tag{19}
$$
于是可得：
$$
f(x)=y'=C_1+C_2e^{-x}\quad F(x)=C_1x-C_2e^{-x}+C_3\tag{20}
$$
此处$C_1,C_2,C_3$都是常数，接下来就是要求解这三个常数的具体值。

将式$(20)$的结果代回到式$(16)$，可得：
$$
C_1+C_2e^{-x}=1+x+\left(\frac12C_1+(1-e^{-0.5})C_2\right)+\left(\frac12C_1-e^{-0.5}C_2+C_3\right)-(C_1x-C_2e^{-x}+C_3)\tag{21}
$$
化简可得：
$$
C_1=1+(1-C_1)x+(C_1+(1-2e^{-0.5})C_2)\tag{22}
$$
显然$C_1=1$，代入后可得$C_2=1/(2e^{-0.5}-1)$，于是我们得到了$f(x)$与$F(x)$的解析表达式：
$$
\begin{aligned}
f(x)&=1+\frac{e^{-x}}{2e^{-0.5}-1}\\
F(x)&=x-\frac{e^{-x}}{2e^{-0.5}-1}+C_3
\end{aligned}\tag{23}
$$
注意到式$(8)$的期望值$T$是式$(12)$中积分面积$S$的两倍，于是有：
$$
T=2S=2[F(1/2)-F(0)]=1+\frac{2-2e^{-0.5}}{2e^{-0.5}-1}=\frac{1}{2e^{-0.5}-1}\approx 4.69348449872319\tag{24}
$$

命题得证。

$\text{Q.E.D.}\blacksquare$""",
# ----
r"""
<font color=red>**证明**</font>：**立体数学归纳法**（我觉得这个名称特别适合这个证明的形式）。

这里还是先定义$a_i$（$i=1,2,...,k$）表示$U$中取值为$i$的元素数量（即每种花色对应的手牌数量），不妨设$a_1\ge a_2\ge...\ge a_k\ge1$

定义$h(a_1,a_2,...,a_k)$表示$U$在最优策略下的期望**盗书**成功次数（$X$的期望），即最大**盗书**成功次数。

根据条件期望的知识，我们有：
$$
h(a_1,a_2,...,a_k)=\frac{a_1}{a_1+a_2+...+a_k}(h(a_1-1,a_2,...,a_k)+1)\tag{15}
$$
式$(15)$显然是正确的，注意式$(15)$非常的重要，这将是下面数学归纳法的根基所在。

- 考察$k=2$的情况，即手牌只包含两种花色的情况：
  $$
  f(U)=h(a_1,a_2)=\frac{a_1}{a_1+a_2}(h(a_1-1,a_2)+1)\\
  g(U)=\frac{a_1}{a_1+a_2}h(a_1-1,a_2)+\frac{a_2}{a_1+a_2}h(a_1,a_2-1)\tag{16}
  $$
  其中$g(U)$的第一项表示拆到了取值为$1$的元素，第二项表示拆到了取值为$2$的元素。则等价证明推导：
  $$
  \begin{aligned}
  f(U)\le g(U)&\Longleftrightarrow\frac{a_1}{a_1+a_2}(h(a_1-1,a_2)+1)\le\frac{a_1}{a_1+a_2}h(a_1-1,a_2)+\frac{a_2}{a_1+a_2}h(a_1,a_2-1)\\
  &\Longleftrightarrow h(a_1,a_2-1)\ge\frac{a_1}{a_2}
  \end{aligned}\tag{17}
  $$
  即等价于证明$h(a_1,a_2-1)\ge a_1/a_2$，做差量代换$a_1=a_2+t$，其中$t\ge 0$

  先考察$a_1=a_2$的情形（$t=0$），即证明$h(a_2,a_2-1)\ge 1$，基于如下的观察：
  $$
  \begin{aligned}
  h(1,1)=1&\Longrightarrow h(1,2)=h(2,1)=\frac23(h(1,1)+1)=\frac43>1\\
  &\Longrightarrow h(2,2)=\frac{2}{4}(h(1,2)+1)>\frac12(1+1)=1\\
  &\Longrightarrow h(2,3)=h(3,2)=\frac{3}{5}(h(2,2)+1)>\frac35(1+1)>1\\
  &\Longrightarrow h(3,3)=\frac{3}{6}(h(2,3)+1)>\frac12(1+1)=1\\
  &...\\
  &\Longrightarrow h(a_2-1,a_2-1)>1\\
  &\Longrightarrow h(a_2,a_2-1)=\frac{a_2}{2a_2-1}(h(a_2-1,a_2-1)+1)>\frac{2a_2}{2a_2-1}>1
  \end{aligned}\tag{18}
  $$
  式$(18)$中的推导本质上是数学归纳法，这样我们就证明了任意的$a_2$，都有$h(a_2,a_2-1)>1$

  那么再从$h(a_2,a_2-1)$作为起点，类似地可以归纳到$h(a_2+t,a_2-1)$：
  $$
  h(a_2+t,a_2-1)=\frac{a_2+t}{2a_2+t-1}(h(a_2+t-1,a_2-1)+1)\ge\frac{a_2+t}{2a_2+t-1}\left(\frac{a_2+t-1}{a_2}+1\right)=\frac{a_2+t}{a_2}\tag{19}
  $$
  式$(19)$中的$\ge$号使用的是已知$(h(a_2+t-1,a_2-1)+1)\ge (a_2+t-1)/a_2$的归纳结果。

  于是$h(a_1,a_2-1)\ge a_1/a_2$总是成立，根据式$(17)$，可知$k=2$时**定理$3$** 成立。

- 接下来考察$k>2$的情况：

$$
  f(U)=h(a_1,a_2,...,a_k)=\frac{a_1}{\sum_{i=1}^k a_i}(h(a_1-1,a_2,...,a_k)+1)\\
g(U)=\frac{a_1}{\sum_{i=1}^k a_i}h(a_1-1,a_2,...,a_k)+\frac{a_2}{\sum_{i=1}^k a_i}h(a_1,a_2-1,...,a_k)+...+\frac{a_k}{\sum_{i=1}^k a_i}h(a_1,a_2,...,a_{k}-1)\tag{20}
$$

类似式$(17)$作等价证明代换：
$$
  \begin{aligned}
  f(U)\le g(U)&\Longleftrightarrow a_2h(a_1,a_2-1,...,a_k)+...+a_kh(a_1,a_2,...,a_{k}-1)\ge a_1
  \end{aligned}\tag{21}
$$
  看起来式$(21)$比式$(17)$要复杂不少，但证明思路是完全一样的，也是使用**双重数学归纳**：

  - 首先说明$f(1,1,...,1)\ge 1/(k-1)$成立。（<font color=red>括号中有$k$个$1$，以$f(1,1)=1$为基，利用式$(15)$直接数学归纳易证</font>）

  - 接下来类似式$(18)$，推出在$a_1=a_2=...=a_k$的情况下，式$(21)$成立，即有：
	$$
	f(a_k,a_k,...,a_k-1)\ge\frac{1}{k-1}\tag{22}
	$$
	注意为什么式$(18)$能够一路推下来，是因为每次推导括号外的乘数必然大于$1/2$，这里类似地乘数必然大于$1/k$，因此可以递推归纳。

  - 最后做类似式$(19)$的归纳证明，得到任意的$a_1\ge a_2\ge ...\ge a_k$都使得式$(21)$的结论成立：

	<font color=red>这个归纳太长，限于篇幅我就不写了，简单地说需要先固定$a_2,...,a_k$来归纳$a_1$，然后固定$a_3,...,a_k$归纳$a_2$，以此类推直到归纳完$a_k$，这也就是为什么称这个证明叫**立体数学归纳法**。</font>

$\text{Q.E.D.}\blacksquare$
""",
# ----
r"""
> <font color=red>**定义$1$（辅助分布列）**</font>：
>
> 已知**可重集**（即可以包含重复元素的集合）$U=\{x_1,x_2,...,x_n\}$中包含$n$个元素，其中$x_i\in\{1,2,...,k\},i=1,2,...,n$。从$U$中**不放回地**依次采样一个元素，并猜测本次采样得到的元素对应的数值，直到猜测错误为止。
>
> 给定一个猜测序列$G$（即$U$中所有元素的一个排列，如$1\rightarrow2\rightarrow1\rightarrow2\rightarrow3$表示为$\{1,2,1,2,3\}$），则$G$对应的**辅助分布列**$[p_1,p_2,...,p_n]$定义为：
> $$
> p_i=\left\{\begin{aligned}
> &\Pr(猜对G中第1个元素)&&(i=1)\\
> &\Pr(猜对G中第i个元素|猜对G中前(i-1)个元素)&&(i=2,...,n)
> \end{aligned}\right.\tag{4}
> $$
> 或者更标准地，定义事件$T_i$表示猜对$G$中第$i$个元素，则：
> $$
> p_i=\left\{\begin{aligned}
> &\Pr(T_1)&&(i=1)\\
> &\Pr(T_i|T_1,T_2,...,T_{i-1})&&(i=1,2,...,n)
> \end{aligned}\right.\tag{5}
> $$

具体而言，仍然以$U=\{1,1,2,2,3\}$为例，按照$1\rightarrow2\rightarrow1\rightarrow2\rightarrow3$的顺序猜测，即$G=\{1,2,1,2,3\}$，辅助分布列为：

|   $p_1$    |   $p_2$   |   $p_3$   |   $p_4$   | $p_5$ |
| :--------: | :-------: | :-------: | :-------: | :---: |
| $\frac 25$ | $\frac24$ | $\frac13$ | $\frac12$ |  $1$  |

这里笔者特意没有给分子分母进行约分，为的就是更清晰地看出分子分母的含义：

- $p_1=2/5$：猜测的是$1$，分子表示此时$U$中还有$2$个$1$，分母表示此时$U$中还有$5$个元素；
- $p_2=2/4$：猜测的是$2$，分子表示此时$U$中还有$2$个$2$，分母表示此时$U$中还有$4$个元素；
- $p_3=1/3$：猜测的是$1$，分子表示此时$U$中还有$1$个$1$，分母表示此时$U$中还有$3$个元素；
- $p_4=1/2$：猜测的是$2$，分子表示此时$U$中还有$1$个$2$，分母表示此时$U$中还有$2$个元素；
- $p_5=1/1$：猜测的是$3$，分子表示此时$U$中还有$1$个$3$，分母表示此时$U$中还有$1$个元素；<font color=red>任何情况下辅助分布列中的最后一个数值都是$1$</font>

那么根据辅助分布列可以快速求得分布列的数值，具体如下面的定理所述：

> <font color=red>**定理$1$（根据辅助分布列求解分布列）**</font>：
>
> 在**问题$1$** 的条件下，已知猜测序列$G$的**辅助分布列**为$[p_1,p_2,...,p_n]$，则猜测序列$G$的分布列为：
> $$
> \begin{aligned}
> &X&\text{Probability}\\
> &0&1-p_1\\
> &1&p_1(1-p_2)\\
> &2&p_1p_2(1-p_3)\\
> &3&p_1p_2p_3(1-p_4)\\
> &...&...\\
> &n-1&p_1p_2...p_{n-1}(1-p_{n})\\
> &n&p_1p_2....p_{n-1}p_n
> \end{aligned}\tag{6}
> $$
> 证明略。

注意到，因为$p_n\equiv1$，因此$X=n-1$的概率总是为零，这与$1.1$节中的结果相吻合。直观上不可能猜错最后一张手牌（此时必然是一张明牌），因此不可能只少猜对一张牌。

**定理$1$** 非常简单，因而省略证明，但是它能够提供$\mathbb{E}_{\rm base}(X)$关于**辅助分布列**的简明表达式：
$$
\mathbb{E}_{\rm base}(X)=p_1+p_1p_2+p_1p_2p_3+...+p_1p_2...p_n=\sum_{i=1}^n\left(\prod_{k=1}^i p_k\right)\tag{7}
$$
这在下面的证明中将非常重要。

----

### 1.3 最优策略的证明

现在我们可以来说明<font color=red>每次猜测集合$U$中包含数量最多的那种元素</font>是明牌条件下的**最优策略**。

> <font color=red>**定理$2$（明牌条件下的盗书最优策略）**</font>：
>
> **问题$1$** 的最优策略是每次猜测集合$U$中包含数量最多的那种元素（如果有多种元素数量同为最多，则任意挑选其中一种即可）。

<font color=red>**证明**</font>：反证法。

定义$a_i$（$i=1,2,...,k$）表示$U$中取值为$i$的元素数量（即每种花色对应的手牌数量）。

不妨设$a_1\gt a_2\ge a_3\ge...\ge a_n$，为了方便说明，这里假定包含数量最多的那种元素是唯一的（即$a_1$严格大于$a_2$），若有多种元素数量同为最多，证明过程是相仿的。

对于某个猜测序列$G=\{x_{g_1},x_{g_2},...,x_{g_n}\}$，假定其中第一次出现$1$的位置是在$x_{g_m}$，即有：
$$
x_{g_1}\neq1,\quad x_{g_2}\neq 1,\quad ...,\quad x_{g_{m-1}}\neq1,\quad x_{g_{m}}=1\tag{8}
$$
此时新构造的猜测序列$G'=\{1,x_{g_1},x_{g_2},...,x_{g_{m-1}},x_{g_{m+1}},...,x_{g_{n}}\}$（即优先猜测包含数量最多的那种元素），利用式$(7)$，我们试图证明，根据$G'$计算得到的$\mathbb{E}'_{\rm base}(X)$总是不小于根据$G$计算得到的$\mathbb{E}_{\rm base}(X)$：

- 设$G$与$G'$的**辅助分布列**分别为$[p_1,p_2,...,p_n]$与$[p_1',p_2',...,p_n']$，定义$f_G(U)$与$f_{G'}(U)$分别表示在猜测序列$G$与$G'$下的猜对次数的数学期望。

- 显然，从第$m+1$次猜测往后，$G$与$G'$是完全都是相同的，即有：
  $$
  p_i=p_i'\quad \forall i=m+1,m+2,...,n\tag{9}
  $$

- 对于每个$i\le m$，根据$1.2$节中描述的**辅助分布列**的计算规律，可以得到形式：
  $$
  \begin{aligned}
  &p_1=\frac{y_1}{n}&&p_1'=\frac{a_1}{n}\\
  &p_2=\frac{y_2}{n-1}&&p_2'=\frac{y_1}{n-1}\\
  &p_3=\frac{y_3}{n-2}&&p_3'=\frac{y_2}{n-2}\\
  &...&&...\\
  &p_{m-1}=\frac{y_{m-1}}{n-m+2}&&p_{m-1}'=\frac{y_{m-2}}{n-m+2}\\
  &p_{m}=\frac{a_1}{n-m+1}&&p_m'=\frac{y_{m-1}}{n-m+1}\\
  \end{aligned}\tag{10}
  $$
  其中$y_1,...,y_{m-1}$表示对应当前猜测时，$U$中还剩下的$x_{g_{1}},...,x_{g_{m-1}}$的数量，显然有$y_i<a_1$（$i=1,2,...,m-1$）成立。

- 根据式$(9)$与式$(10)$的规律，代入到式$(7)$中可得$G$与$G'$对应的数学期望值：
  $$
  \mathbb{E}_{\rm base}(X)=f_G(U)=p_1+p_1p_2+...+p_1p_2...p_m(1+p_{m+1}+p_{m+1}p_{m+2}+...+p_{m+1}p_{m+2}...p_{n})\\
  \mathbb{E}_{\rm base}'(X)=f_{G'}(U)=p_1'+p_1'p_2'+...+p_1'p_2'...p_m'(1+p_{m+1}'+p_{m+1}'p_{m+2}'+...+p_{m+1}'p_{m+2}'...p_{n}')\tag{11}\\
  $$
  显然式$(11)$中两行各自的最后一项括号中的部分完全相等。

  因为$a_1$比每一个$y_i$（$i=1,2,...,m-1$）都要严格地大，容易发现有：
  $$
  \begin{aligned}
  &\frac{a_1}{n}>\frac{y_1}{n}&&\Longrightarrow p_1'>p_1\\
  &\frac{a_1y_1}{n(n-1)}>\frac{y_1y_2}{n(n-1)}&&\Longrightarrow p_1'p_2'>p_1p_2\\
  &\frac{a_1y_1y_2}{n(n-1)(n-2)}>\frac{y_1y_2y_3}{n(n-1)(n-2)}&&\Longrightarrow p_1'p_2'p_3'>p_1p_2p_3\\
  ...&&...\\
  &\frac{a_1y_1y_2...y_{m-2}}{n(n-1)...(n-m+2)}>\frac{y_1y_2y_3...y_{m-1}}{n(n-1)...(n-m+2)}&&\Longrightarrow p_1'p_2'...p_{m-1}'>p_1p_2...p_{m-1}\\
  &\frac{a_1y_1y_2...y_{m-2}y_{m-1}}{n(n-1)...(n-m+1)}=\frac{y_1y_2y_3...y_{m-1}a_1}{n(n-1)...(n-m+1)}&&\Longrightarrow p_1'p_2'...p_{m-1}'p_{m}'=p_1p_2...p_{m-1}p_m\\
  \end{aligned}\tag{12}
  $$
  即前$m-1$项都是$\mathbb{E}_{\rm base}'(X)$严格地大于$\mathbb{E}_{\rm base}'(X)$中的对应项，最后一项括号外的部分完全相等。

  于是有：
  $$
  \mathbb{E}_{\rm base}'(X)>\mathbb{E}_{\rm base}(X)\tag{13}
  $$

综上所述，上面的证明论述了这样的一个事实：

- 对于猜测序列$G$，只要它在**当次猜测时**没有选取此时集合$U$中包含数量最多的那种元素，那我们总是可以重新构造一个新的猜测序列$G'$满足**当次猜测时**选取此时集合$U$中包含数量最多的那种元素（即把该元素直接调换到序列开头），并且利用$G'$序列进行猜测的猜对次数的数学期望比$G$严格地大。这样就说明了<font color=red>每次猜测集合$U$中包含数量最多的那种元素</font>是最优策略。

$\text{Q.E.D.}\blacksquare$
""",
# ----
r"""
# 高级运筹与优化理论 Homework 5

## Problem 5.1

- $(a)$ 可行集（*feasible set*）为闭区间$[2,4]$，因为目标函数$x^2+1$是单调在可行集上单调递增，因此在最优解$x^*=2$（*optimal solution*）时取得最优值（*optimal value*）$p^*=5$；

- $(b)$ 目标函数与可行集的绘图如$\rm Figure\space 1$所示（最优点与最优值标注在图中）：

  ![Figure 1](1.png)

  拉格朗日函数（*Lagrangian*）为：
  $$
  L(x,\lambda)=1+x^2+\lambda(x-2)(x-4)=(1+\lambda)x^2-6\lambda x+(1+8\lambda)\tag{1.1}
  $$
  分别代入$\lambda=0,1,2,3$可得拉格朗日函数在若干不同$\lambda$值下的图像，如$\rm Figure\space 2$所示：

  ![Figure 2](2.png)

  若$\lambda\le-1$，则显然拉格朗日函数无界；若$\lambda>-1$，则拉格朗日函数在$\tilde x=\frac{3\lambda}{(1+\lambda)}$取得最小值；可得：
  $$
  g(\lambda)=\left\{\begin{aligned}&-\lambda+10-\frac{9}{1+\lambda}&\lambda>-1\\&-\infty\quad&\lambda\le-1\end{aligned}\right.\tag{1.2}
  $$
  $g(\lambda)$的图像如$\rm Figure\space3$所示：

  ![Figure 3](3.png)

  根据$\rm Figure\space 3$容易看出$g(\lambda)$在$\lambda^*=2$时取得最大值$g(\lambda^*)=5$，可得$p^*\ge\inf_x L(x,\lambda),\lambda\ge0$，即下界性质成立；

- $(c)$ 拉格朗日对偶问题如下所示：
  $$
  \begin{aligned}
  \text{maximize}\quad
  &-\lambda+10-\frac{9}{1+\lambda}\\
  \text{subject to}\quad
  &\lambda\ge0
  \end{aligned}\tag{1.3}
  $$
  显然目标函数的二阶导数为$g''(\lambda)=-\frac{18}{(1+\lambda)^3}<0$（$\lambda\ge0$），则对偶问题的目标函数是凹函数；

  对偶问题在最优解$\lambda^*=2$时取得最优值$d^*=5$，由于$p^*=d^*=5$，因此强对偶性（*strong duality*）成立；

- $(d)$ $u<-1$时，*perturbed problem*无解（因为$(x-2)(x-4)$在$x=3$时取得最小值$-1$），$p^*(u)=\infty$；

  $u\ge-1$时，*perturbed problem*可行集为$\left[3-\sqrt{1+u},3+\sqrt{1+u}\right]$，以下分两种情况讨论：

  1. 若$-1\le u\le8$，则$x^2+1$在$\left[3-\sqrt{1+u},3+\sqrt{1+u}\right]$上单调递增，因此最优解$x^*(u)=3-\sqrt{1+u}$，此时最优值$p^*(u)=u-6\sqrt{1+u}+11$；
  2. 若$u\ge8$，则$x^2+1$的全局最优解$0\in\left[3-\sqrt{1+u},3+\sqrt{1+u}\right]$，因此最优解$x^*(u)=0$，此时最优值$p^*(u)=1$；

  综上所述，可得$p^*(u)$的表达式为：
  $$
  p^*(u)=\left\{\begin{aligned}
  &\infty\quad&u<-1\\
  &u-6\sqrt{1+u}+11\quad&-1\le u\le8\\
  &1\quad&u\ge8
  \end{aligned}\right.\tag{1.4}
  $$
  函数$p^*(u)$的图像如$\text{Figure 4}$所示：

  ![Figure 4](4.png)

  从图像上可以看出$\frac{\text{d}p^*(0)}{\text{d}u}=-\lambda^*=-2$，也可以解析求解：
  $$
  \frac{\text{d}p^*(0)}{\text{d}u}=\left.\frac{\text{d}p^*(u)}{\text{d}u}\right|_{u=0}=\left.1-\frac{3}{\sqrt{1+u}}\right|_{u=0}=1-3=-2\tag{1.5}
  $$

- 

## Problem 5.4

- $(a)$ 这等价于求解线性规划：
  $$
  \begin{aligned}
  &\text{minimize}\quad c^\top x\\
  &\text{subject to}\quad w^\top Ax\le w^\top b
  \end{aligned}\tag{2.1}
  $$
  不妨设$\left\|w^\top A\right\|_2=w^\top AA^\top w=1$，否则我们总是可以在式$(2.1)$的约束两侧同除以$\left\|w^\top A\right\|_2$即可得到正则化的形式；

  注意到式$(2.1)$中的线性规划总是有解，因为向量$c$总是可以分解成两个向量（分别与$w^\top A$正交和平行）：
  $$
  c=\lambda A^\top w+\hat c\tag{2.2}
  $$
  其中$\hat c^\top A^\top w=0$；针对式$(2.2)$中的分解形式做如下讨论：

  1. 若$\lambda>0$，令$x=-tA^\top w$，则有：
	 $$
	 c^\top x=-tc^\top A^\top w=-t\left(\lambda w^\top A+\hat c^\top\right)A^\top w=-t\lambda w^\top AA^\top w=-t\lambda\overset{t\rightarrow+\infty}{\longrightarrow}-\infty\tag{2.3}
	 $$
	 且当$t\rightarrow+\infty$时约束总是成立：
	 $$
	 w^\top Ax-w^\top b=-tw^\top AA^\top w-w^\top b=-t-w^\top b\le0\tag{2.4}
	 $$
	 因此式$(2.3)$中的线性规划是无界的，即$\inf_{x\in H_w}c^\top x=-\infty$；

  2. 若$\hat c\neq0$，令$x=w^\top bA^\top w-t\hat c$，则有：
	 $$
	 c^\top x=\left(\lambda w^\top A+\hat c^\top\right)\cdot\left(w^\top bA^\top w-t\hat c\right)=\lambda w^\top b\cdot w^\top AA^\top w-t\hat c^\top\hat c=\lambda w^\top b-t\hat c^\top\hat c\overset{t\rightarrow+\infty}{\longrightarrow}-\infty\tag{2.5}
	 $$
	 且当$t\rightarrow+\infty$时约束总是成立：
	 $$
	 w^\top Ax-w^\top b=w^\top A\left(w^\top bA^\top w-t\hat c\right)-w^\top b=w^\top b\cdot w^\top AA^\top w-w^\top b=0\tag{2.6}
	 $$
	 因此式$(2.3)$中的线性规划是无界的，即$\inf_{x\in H_w}c^\top x=-\infty$；

  3. 若$\hat c=0$（即$c=\lambda A^\top w$）且$\lambda\le 0$，根据$w^\top Ax\le w^\top b$，可知最优值显然为$\lambda w^\top b$；

  综上所述，我们有：
  $$
  \inf_{x\in H_w}c^\top x=\left\{\begin{aligned}
  &\lambda w^\top b&\quad\text{if }c=\lambda A^\top w\text{ for some }\lambda\le0\\
  &-\infty&\quad\text{otherwise}
  \end{aligned}\right.\tag{2.7}
  $$
  
- $(b)$ 根据$(a)$中的结论，可以对应地写出规划问题：
  $$
  \begin{aligned}
  &\text{maximize}\quad\lambda w^\top b\\
  &\text{subject to}\quad c=\lambda A^\top w\\
  &\quad\quad\quad\quad\quad\lambda\le0,w\succeq0
  \end{aligned}\tag{2.8}
  $$
  注意到约束$c=\lambda A^\top w$不是非凸，因此式$(2.8)$不是凸规划。

- $(c)$ 首先写出式$(2.1)$的对偶问题（决策变量$y$为标量，因为原问题只有一个约束）：
  $$
  \begin{aligned}
  &\text{maximize}\quad yw^\top b\\
  &\text{subject to}\quad yA^\top w=c\\
  &\quad\quad\quad\quad\quad y\le0
  \end{aligned}\tag{2.9}
  $$
  然后在式$(2.8)$中令$z=-\lambda w$代入，可得：
  $$
  \begin{aligned}
  &\text{maximize}\quad-b^\top z\\
  &\text{subject to}\quad A^\top z+c=0\\
  &\quad\quad\quad\quad\quad z\succeq0
  \end{aligned}\tag{2.10}
  $$
  对比式$(2.9)$与式$(2.10)$可以发现，令$z=-yw$，则两式完全等价。

## Problem 5.12

根据提示，设$y_i=b_i-a_i^\top x$，则原问题可以表示为：
$$
\begin{aligned}
&\text{minimize}\quad-\sum_{i=1}^m\log y_i\\
&\text{subject to}\quad y=b-Ax
\end{aligned}\tag{3.1}
$$
其中：
$$
y=\left[\begin{matrix}y_1&y_2&...&y_m\end{matrix}\right]^\top,b=\left[\begin{matrix}b_1&b_2&...&b_n\end{matrix}\right]^\top,A=\left[\begin{matrix}a_1^\top\\a_2^\top\\...\\a_m^\top\end{matrix}\right]_{m\times n}\tag{3.2}
$$
根据式$(3.1)$可得拉格朗日函数：
$$
L(x,y,\lambda)=-\sum_{i=1}^m\log y_i+\lambda^\top\left(y-b+Ax\right)\tag{3.3}
$$
根据式$(3.3)$可以写出对偶函数：
$$
g(\lambda)=\inf_{(x,y):y\succ0,Ax\prec b}\left(-\sum_{i=1}^m\log y_i+\lambda^\top\left(y-b+Ax\right)\right)\tag{3.4}
$$

1. 若$\lambda\not\succ0$，则$\lambda$至少有一个分量$\lambda_i\le0$，于是总是可以令对应的$y_i$取任意大的正数，从而使得$L(x,y,\lambda)$趋于负无穷；
2. 若$\lambda\succ0$且$\lambda^\top A\neq 0$，则$\lambda$每个分量$\lambda_i>0$，此时总是可以取$x$使得$\lambda^\top Ax$趋于负无穷；
3. 若$\lambda\succ0$且$\lambda^\top A=0$，则$L(x,y,\lambda)=-\lambda^\top b+\sum_{i=1}^m\left(\lambda_iy_i-\log y_i\right)$，在$y_i=\frac1{\lambda_i}$时取得最小值；

综上所述，可得$g(\lambda)$的表达式：
$$
g(\lambda)=\left\{\begin{aligned}
&\sum_{i=1}^m\log\lambda_i+m-\lambda^\top b\quad&\lambda\succ0\text{ and }\lambda^\top A=0\\
&-\infty\quad&\text{otherwise}
\end{aligned}\right.\tag{3.5}
$$
根据式$(3.5)$可以写出对偶问题：
$$
\begin{aligned}
&\text{maximize}\quad\sum_{i=1}^m\log\lambda+m-\lambda^\top b\\
&\text{subject to}\quad A^\top\lambda=0,\lambda\succ0
\end{aligned}\tag{3.6}
$$

## Problem 5.31

证明：

注意到原问题是凸规划，于是$f_0,f_1,f_2,...,f_m$都是凸函数，因此对于任意可行的$x$，有$\nabla^2f_i(x)\succeq0$，于是$\forall i$，我们有：
$$
0\ge f_i(x)=f_i(x^*)+\nabla f_i(x^*)^\top(x-x^*)+\frac12(x-x^*)^\top\nabla^2f(\zeta)(x-x^*)\ge f_i(x^*)+\nabla f_i(x^*)^\top(x-x^*)\tag{4.1}
$$
其中$x^*$为最优解，$\zeta$是位于$x$与$x^*$之间的某个向量；

根据$\lambda^*_i\ge0$与式$(4.1)$，结合**KKT**条件中的互补松弛和拉格朗日函数一阶导为零两个等式条件，可得：
$$
\begin{aligned}
0&\ge\sum_{i=1}^m\lambda_i^*\left(f_i(x^*)+\nabla f_i(x^*)^\top(x-x^*)\right)\\
&=\sum_{i=1}^m\lambda_i^*f_i(x^*)+(x-x^*)\sum_{i=1}^m\lambda_i^*\nabla f_i(x^*)^\top\\
&=(x-x^*)\cdot\left[-\nabla f_0(x^*)^\top\right]\\
&=-\nabla f_0(x^*)^\top(x-x^*)
\end{aligned}\tag{4.2}
$$
由式$(4.2)$即有$\nabla f_0(x^*)^\top(x-x^*)\ge0$，命题得证；

证毕。$\blacksquare$

## Question 5

> $x^{(3)}$是最优解，其他两个不是。

拉格朗日函数及其一阶偏导为：（其中$x_1\ge0,x_2\ge0,\mu_1\ge0,\mu_2\ge0$）
$$
\begin{aligned}
L(x,\mu)&=(x_1-2.25)^2+(x_2-2)^2+\mu_1(x_1^2-x_2)+\mu_2(x_1+x_2-6)\\
&=(1+\mu_1)x_1^2+x_2^2+(\mu_2-4.5)x_1+(\mu_2-\mu_1-4)x_2+(9.0625-6\mu_2)\\
\nabla_{x_1} L(x,\mu)&=2(1+\mu_1)x_1+\mu_2-4.5\quad \nabla_{x_2} L(x,\mu)=2x_2+\mu_2-\mu_1-4\quad\\
\end{aligned}\\
\tag{5.1}
$$
则原问题的最优解$x^*$与对偶问题的最优解$\mu^*$应当满足**KKT**条件：
$$
\left\{\begin{aligned}
&{x_1^*}^2-x_2^*\le0\quad&\\
&{x_1^*}+x_2^*-6\le0\quad&\\
&x_i^*\ge0\quad&i=1,2\\
&\mu_j^*\ge0\quad&j=1,2\\
&2(1+\mu_1^*)x_1^*+\mu_2^*-4.5=0\\
&2x_2^*+\mu_2^*-\mu_1^*-4=0\\
&\mu_1^*({x_1^*}^2-x_2^*)=0\\
&\mu_2^*(x_1^*+x_2^*-6)=0\\
\end{aligned}\right.\tag{5.2}
$$
下面分别代入$x^{(1)},x^{(2)},x^{(3)}$到**KKT**条件中：

1. 代入$x_1^*=2.25,x_2^*=2$；

   显然原问题不可行，则不是最优解；

2. 代入$x_1^*=0,x_2^*=2$；

   根据${x_1^*}^2-x_2^*\neq0$，可得$\mu_1^*=0$；根据${x_1^*}+x_2^*-6\neq0$，可得$\mu_2^*=0$；

   此时容易发现$(x_1^*,x_2^*,\mu_1^*,\mu_2^*)$不满足式$(5.2)$中第五行约束，即$2(1+\mu_1^*)x_1^*+\mu_2^*-4.5\neq0$，因此$x^{(2)}$不是原问题的最优解；

3. 代入$x_1^*=1.5,x_2^*=2.25$；

   根据${x_1^*}^2-x_2^*=0$，可得$\mu_1^*\ge0$；根据${x_1^*}+x_2^*-6\neq0$，可得$\mu_2^*=0$，

   将结果代入到式$(5.2)$中第五行和第六行，可得：
   $$
   \left\{\begin{aligned}
   &2\cdot(1+\mu_1^*)\cdot1.5+0-4.5=0\\
   &2\cdot2.25+0-\mu_1^*-4=0
   \end{aligned}\right.\tag{5.3}
   $$
   发现式$(5.3)$存在唯一解$\mu_1^*=0.5\ge0$；

   容易发现$x_1^*=1.5,x_2^*=2.25,\mu_1^*=0.5,\mu_2^*=0$满足式$(5.2)$中所有约束，因此$x^{(3)}$是最优解；
""",
# ----
"""“大成报，拒了。”　　

“大河报，又拒了。”　　

“风华报，又又拒了。”　　

“大宫报，说得这么委婉，就是不要。”　　

“小风报，竟然让我写《书剑风华录》，疯了吧这是。要是名报的景明镛发飙，老板跑了，我往哪里跑。我真跑了，小犹太怎么办？”　　
看着诸多小报的回复，赵正忍不住一边吐槽，一边看信。

上辈子作为997福报的非知名网站小编，回到这85年的港城，刚被某大型报社辞退，干啥啥不行，他能怎么办。

除了修长的身材和这张比华仔、黎仔还帅的脸蛋，简直是一无是处。

难道，让他去混社团？

扑街啦，那种天天被阿sir查身份证、什么时候去地府报道的日子，赵正可不想过。

而且，一般不拿西瓜刀从九龙砍到湾仔的古惑仔，就帮别人停停车，根本赚不到钱。

目前来看，还是当文抄公，嗯，当作家、当知识分子更赚钱。　　

某些报纸的签约作家，三天打鱼，两天撒网，一周更个万把字，就能拿到上万的月薪，让人眼红。　　

可惜，现在的港城小报编辑一点都不识货，赵正写了几本小说的开头，比如退婚、莫欺少年穷、斩十四境大妖，都被毙了。

这群扑街仔，眼光简直是扑你老木啊！　　

“我要给你我的追求，还有我的自由。可你却总是笑我，一无所有...”　　

“阿正，吃饭啦。你刚刚说什么，一无所有啊？”　　

正当赵正为生计发愁的时候，一个娇俏可人的身影出现在房门口，朝着他喊了一句，脸上的笑容如春天般抚慰着男人的心灵。　

只是一身简单的牛仔和白色短袖，就能让人双眼发亮，脸上的笑容能让男人的烦心事尽数消散，称得上一句‘青春无敌、玉女掌门’。

“来了。”　　

看着小犹太的甜美笑容，赵正微笑着起身走了过去。

小犹太的家就住在赵正对门，比他小三岁，从小就跟在他身后瞎跑，算是他的青梅竹马。

两人一起从小学读到中学，而赵正勉强考上了港大，却因父母出事，学费没有着落而辍学；周蕙慜则是因为文化课成绩不行，读完中学六年，就去电台上班，补贴家用，供养身体不太好的母亲。

总的来说，两家的家境差不多，唯一的区别，赵正的600尺唐楼（60平老房子）是父母留的，周家母女的则是租的。

来到对门的一居室，赵正就看到坐在沙发上等候的周母，脸色有些不善。　　

换做哪一个母亲，见到女儿和没什么前途的小年轻处朋友，都不会有什么好脸色。

那张脸好看一点有什么用，长得比发哥、华仔还帅有什么用，能当饭吃吗？

好好的大报社工作不干，一天天地做什么大作家的美梦，天天在她家混吃混喝，除了对门那套小房子和脸蛋真是一无是处。

“伯娘（伯母），下午好。”

知道这位周母的心思，赵正也没在意，微笑着打了招呼。　　

毕竟，拱了对方辛苦养大的好白菜，赵正总不能还要求对方带什么好脸色。

或许他该向现实妥协了，回头就写本‘风月无边’的小说，去小风报那边谈谈价钱。

为了钱妥协，不丢人。　　

看着身边盛汤的小犹太，赵正觉得，为了爱情都是值得的。　　

此时尚未出道的王霏，多年后写了一首歌，为了爱情...3　　

“妈，喝汤。”　　

给妈妈盛了一碗汤后，周蕙慜喊了一声，免得对方说出什么不合适的话。　　

在这个家里，她现在已经是赚钱的顶梁柱，妈妈也得给她几分面子。　　

“好。”　　

点了点头，知道女儿胳膊肘往外拐的周母，默默开始喝汤。　　

“阿正，喝汤。”　　

见妈妈没有再说，周蕙慜开心地给男朋友也盛了一碗。　　

虽然阿正没有跟她正式表白，但是周蕙慜早已经把对方当成了另一半。　　

女孩子认定了一个男孩子，就要一辈子，此生不渝。　

“阿正，你写书写得咋样了？咳咳。”　

快吃完饭的时候，周母开启了新的话题。　　

既然女儿认定了对方，她也不好强行拆散，总得让未来女婿找个活干，再不济去餐馆端盘子，也有个三千一个月，可以补贴家用。　　

工作早点稳定下来，两人也可以住到一起，卖了老房子换套新房，省得多租一间房子。　　

勤俭持家，才能越来越好。　　

“妈。”　　

听到妈妈旧事重提，周蕙慜忍不住打断了对方的话。　　

从小就和阿正相识相知，她可是知道对方的才华。　　而且，周蕙慜先前看过男朋友写的手稿，故事和主角都很吸引人，她都想知道后续的小说情节，只是那些报社的编辑不识货。　　

若是她男朋友的书写得不好，怎么会有某些的大报社主编贪得无厌，故意压价，还要作者名的全部版权。　　

“明天去签约。”　　

对此，赵正握住了身旁妹子的手，微笑着回答道。　　

这一刻，他算是彻底下定了决心。　　

一个男人，怎么能靠女人养着，他必须要拿回经济主动权。　　

先前那个眼高手低的赵正，已经在被某报社辞退、郁郁而终后没了，此时的他要在这寸土寸金的港城混口饭吃，并不难。　　

“真的？”　　

一听这话，周母眼前一亮，立马开始规划：“你们两个一起工作，存钱一定很快。等存够了新房子的首付，就可以办个婚礼了。”　　

“妈妈。”　　

没想到妈妈说得这么直白，周蕙慜脸色微红地打断，还为男朋友考虑：“阿正有他自己的规划，你别帮他做决定了。”　　

说起这句话的时候，周蕙慜的左手感受到男朋友手心的温热，眼神也是忍不住瞥了过去。　　

虽然男朋友还没有正式告白，但是她在梦里的时候，已经幻想过两人一起走进婚礼殿堂的样子，心里带着憧憬。

“伯娘放心，我一定会让敏敏有个最好的婚礼。”　　

感受到小犹太的眼神，赵正很是肯定地说道，握着对方的右手也是多了几分力道。　　

相识于微末，小犹太没有嫌弃他之前的那般眼高手低，赵正自然也不会让她输。　　

即便他未来的成就再高，家里的那个位置肯定是小犹太的。

“妈妈，我相信阿正。”　　

心里满是甜蜜的周蕙慜，眼里的幸福都快溢出来了。""",
# ----
"""
“嗯，那就好。”

点了点头，周母算是认可了这个小年轻的担当。

不过，一切都要看实际行动。

若是对方一直待在家里，靠她女儿养着，周母肯定不会答应这门婚事。

“谢谢伯娘。”

吃完晚饭，向来身体不太好的周母，就坐在客厅看着黑白电视休息。

而准备帮忙收拾碗筷的赵正，则是被小犹太以‘君子远庖厨’给赶出了厨房。

没过多久，在厨房里忙完的周蕙慜，和妈妈说了一声，就走到对门的房间，轻手轻脚地走进卧室。

来到奋笔疾书的男朋友身旁，周蕙慜小心地将一杯温热的铁观音茶水放到桌上。

“你来了。”1

闻到清香味的赵正，放下手中的笔，笑着将小犹太拉入怀中，感受着凹凸有致的曲线，身体有了很自然的反应。

“阿正，我妈妈说的话，你别多想。写书这种事，讲究的是厚积薄发，你不要急。”

坐在男朋友怀里，周蕙慜主动帮妈妈道歉。

她相信男朋友的才华，自然全力支持对方。

“没事，我已经想好了，明天和小风报的主编谈谈，尽快签约。”

感受到小犹太的心意，赵正主动说起了自己的规划。

对于那位小风报主编要求的文章，他已经有了腹稿，刚刚就已经在写开头。

作为金古以后最有影响力的武侠作家，非黄一莫属。

尤其是那本开创了兼具风月和剧情的《覆雨翻云》，赵正上辈子的高中时期，可是拿着手电筒，趴在学校寝室的被窝里看完的。21

那年少气盛的时代，看完一些章节，大晚上的都和现在这般，一根擎天白玉紫金梁，仿佛要撑起天穹。2

作为‘景顾’两位大家之后的武侠大家，黄一的知名度可谓是七零、八零后两代人中的翘楚，甚至在九零年代，内地满大街都是黄一的小说。27

嗯，全是黄色书封，全是假冒伪劣产品！

只是当时还算个少年的赵正，和其余同龄人一样，惊讶于这位大作家的才华，心头只有‘牛比’两个字。

“这个就是你写的新书吗？”

听到男朋友的话，周蕙慜不疑有他，好奇地看向桌上的新稿子。

“刚写了个开头，你帮我看看。”

为了先声夺人，赵正大致修改了一下开头的情节，正好可以让小犹太帮忙品鉴一下。

原著里的开头，本是浪翻云和上官鹰的帮派争斗，略微显得平淡。

报纸连载和整本书出版，两者之间的区别还是很大的，远远大于几十年后的网络小说和实体小说。

开头黄金三章，尤为关键。

索性，赵正将魔师庞斑隐居前的一段，当做了楔子，稍微改动了一点，后续的情节依旧没动。

大元第一高手对阵大明前·第一高手，外加魔师庞斑利用慈航静斋圣女传人的片段，绝对足够精彩。

“嗯。”

只是，周蕙慜拿起稿子看了几分钟，就感觉心口有些不对。

翘屯的感受，刚才就已经很明显了，但是男朋友像今天这般安慰她的心，还是第一次，让她的脸红彤彤的，脑子都有些晕。

“阿正...”

“叫我干什么？”

手上的太极招式变化，赵正凑到小犹太的耳边，轻声问道。

年轻人血气方刚，他觉得自己下意识的动作可以理解。

“你想的话，我可以的。”

早已经视对方为另一半的周蕙慜，脸上满是红晕地嘟喃道。

虽然她还没有真正体验过，但是读中学的时候，没少听同班的其他女孩说起，自然也懂得天地人伦之道。

“好。”

眼看小犹太都这么说了，赵正也没想着等到明天和小风报签约以后再庆祝。

男人，最忌讳的就是犹犹豫豫。

就像某些港片里的助人情节，某些男主角面对女朋友的投怀送抱，竟然还能坐怀不乱，等到功成名就之后再明媒正娶。

结果呢，那些女主角都会在偶然中遇到什么大坏蛋，丢了清白，之后就是虐心环节。

不得不说，那些编剧可能是把自己的初恋代入到女主角身上，才有那般的感同身受。

为了不虐心，赵正只能顺其自然。

“呀，阿正。”

“敏敏，这辈子，我都认定你了。”

关键的时刻，赵正强压住内心的猛虎，凑到女朋友耳边温柔地说着情话：“我一定给你买套海边的大房子，每天带你看海，看遍春暖花开，朝阳夕落。”

至于期限，赵正没有说。

爱情，本就没有期限。

“嗯，我相信你。”

听着那感人肺腑的情诗，周蕙慜抱着男朋友的脖子，脸上没有丝毫失落，尽是对未来美好的憧憬。

之后的事，自然是雨落峡谷，水到渠成。

“哼。”

而坐在家里客厅看完了华仔和伟仔主演的杨家将，周母看了眼对面紧闭的房门，叹了口气，轻轻掩上了房间门。

女大不中留，希望女儿看男人的眼光没问题。

还别说，在颜值方面，女儿的眼光确实不错，找了个比华仔他们还帅的。

“呼。”

等到怀里折腾了两番的小犹太沉沉睡去，赵正轻手轻脚地起身，简单洗了把脸，就坐在书桌前奋笔疾书。

既然之前在伯娘那边夸下了海口，赵正就要说到做到，尽量把精彩的部分写出来，打动对方，提高卖相，也好开个好价钱。

正好，在他稍微改编之下，覆雨翻云开头就是魔师庞斑为蒙古复国大业算计圣女的情节，刚才脑海里有所印象的赵正，写起来得心应手。

因为太过投入，赵正把自己和小犹太代入进去，不知不觉就来了莫名的感觉。

等到将近六千字的稿子完成，有些口干舌燥的赵正，看了眼熟睡的小犹太，还是不忍心打扰刚刚破瓜的女朋友。

又去洗了个冷水脸，没有丝毫睡意的赵正，继续坐到书桌前开始疾书。

不同于上辈子为了房贷车贷，这辈子身处在80年代的港城，没有钱的话，连个人身安全都保证不了。

997都卷过了，还怕一个通宵写书，开玩笑！！！

正好，这个精神状态，非常适合写擦边情节。

嗯，在后世的某点上说擦边，但是赵正真写出来的，那是非常的大胆，可谓毫无顾忌。

现在港城的报社行业可没有什么‘君子协定’，写纯纯的小颜色文都能堂而皇之地出现在报纸上，更不用说那些没有丝毫底限的二级以下电影。

“应该不会太过吧。”

写完一万字的赵正，揉了揉手腕，趁着休息时间看了下手稿，觉得略微有些脸红。

要是上辈子的网站小编身份，看到这种稿子，分分钟毙掉，顶多放在自己的垃圾箱里，抽空批判一番，哪里知道重生回到港城，还得以这个谋生。

人啊，得学会换位思考。

若是连生存都保障不了，谈什么面子和节操。
""",
# ----
"""
“嗯。”

第二天的清晨，看着熟睡的老公，周蕙慜悄悄起身，去楼下买好了早餐，给妈妈留了份，再回到男朋友的屋里。

轻手轻脚地来到书桌前，周蕙慜看着一小堆的新书稿，知道男朋友昨晚挑灯夜战，在为两人的未来而努力，心里一下子多了几分甜蜜。

只不过，坐下来看了会书稿，周蕙慜品味着里面旖旎的情节，联想到昨晚的一幕幕，脸色忍不住变得红彤彤的。

可是，这个小说情节有些好看，让周蕙慜有些欲罢不能。

“好看吗？”

不知何时，一个声音在周蕙慜耳边响起。

回过神来的周蕙慜，转头看着男朋友的俊脸，笑着说道：“阿正，你醒啦。”

“嗯。”

说着话的时候，赵正的两只手开始不安分起来。

先前不忍打扰刚刚从少女蜕变的女朋友，现在离对方上班时间还有一个半小时，足够了。

“阿正...”

面对男朋友的痴迷，看了小说本就有些情动的周蕙慜，一双明媚的大眼睛拉着丝，转身双手主动抱了上去。

这一轮男女感情的升华，因为周蕙慜初经风雨的稍有不适，慢条斯理地收场了。

“今天身体不舒服，我帮你请一天假。”

等女朋友躺在床上休息，赵正体贴地说了句。

从今天开始，他也能努力赚钱，家庭开支的重担不需要女朋友承担。

“没事的，我刚转正，下午还要录制节目，不太好请假。”

对于请假这种事，想要赚全勤的周蕙慜是拒绝的。

现在，男朋友的稿费还没有着落，她一个月五千多的工资，算是家里的全部收入。

“行，我会尽快拿到稿费。”

知道小犹太的性格，赵正也没有多说，尽早拿到稿费才是正理。

“我相信你，阿正。”

在男朋友怀里蹭了蹭，周蕙慜看了下床头柜上的闹钟，连忙起身坐起：“我等下要去上班了，咱们先去吃早餐。”

男朋友可是要成为大作家的人，在家里休息没事，她只是电台的一个小员工，迟到可是要扣工资的。

在男朋友功成名就之前，周蕙慜需要负担家里的开支，迟到一次被扣全勤，都是一个不能忽视的损失。

再者，这点小伤，克服一下，她在电台演播室又不用时刻走来走去，根本不是问题。

“我帮你洗洗。”

“哎呀，别闹。”

一阵清晨的欢闹过后，赵正送小犹太去了位于九龙马可尼道的港城电台。

今年上半年，周蕙慜并没有参加新秀歌唱大赛，而是直接去了业余DJ大赛，拿到不知名的冠军后去了港城电台工作。

之所以这样的选择，周蕙慜只是想着港城电台的工资高一点，夜班还有加班费，保底5000起步。

“阿正，等下。”

刚要进门的周蕙慜，想起什么，连忙喊住了要离开的男朋友。

“怎么了？”

还以为刚刚进展到最后一步的小犹太舍不得自己，赵正笑着走过去，抱了对方一下。

“等下可能要用到，你先拿着。”

从自己的小包里拿出500港元，周蕙慜生怕男朋友在外面舍不得花钱。

“行，谢谢敏敏。”

看着女朋友如此贴心的模样，先前已经拿过200的赵正，倒是没有拒绝。

等到他开始拿稿费，肯定交给对方保管，没有必要在这电台门口拉拉扯扯。

“预祝阿正成功。”

亲了下男朋友的脸，周蕙慜开心地走进大门，

而赵正也没拖沓，背着一只单肩书包上了电车，往新界方面赶去。

这年头港城的治安不咋滴，出门谈判也尽量别拿公文包，什么时候被古惑仔看到，掏出一把小刀找你化缘都不一定。

这一点，赵正比较谨慎。

因为名报、东华日报、港城星等销量靠前的报纸都在新界方面的大埔，其余的小报社也是跟风在这附近开设档口，只是隔了一两条街。

销量多不多不重要，沾点气运最重要。

早饭的时候，已经打电话和小风报的主编约好，赵正提前半小时踩点了一下小风报的地址和销量。

地址就不说，一个巴掌大的小作坊，可能华仔门徒里的工厂也就差不多大。

此时的港城，普通人没有什么信息接收渠道，除了电视台就只有报纸，因此报纸行业就像三四十年后的非法小网站那么多，那么卷。

而港城有关部门，对这个现象也是睁一只眼闭一只眼，反正这些人不犯法就行。

“500，还退个三四百。”

门口老大爷收下一包健牌香烟，说出的销量数字让赵正大吃一惊。

每天就200的销量，这小风报是怎么活下来的，怎么养得起报社的员工，靠情怀吗？

要知道，前两天名报刚在他们首页上庆祝，日销量超过6万，创始人景明镛刚花了1250万港元买下山顶道的豪宅。

就这200销量，人工费都赚不回来。

“靓仔，我劝你一句。要是来找工作的，换一家嘛。”

看在这10块一包的香烟上，月薪过3000的老大爷也是打开了话茬：“这报纸已经换了三个老板，现在这个原来是做装修。原本嘛，想着靠报纸的销量给他们装修公司免费打个广告，结果越办越差。现在整个报社就三个人，一个做主的，一个打样的，一个印刷的。差不多，也快换老板了。”

说完之后，热情的老大爷主动劝道：“要是想上工，旁边那家雪月报还不错，每天销量两三千，有时候还加印呢。”

“谢谢大爷。”

没有解释什么，赵正走向了不远处的茶楼。

这大埔报社一条街，自然少不了文化人最爱的茶楼，那真算得上十步一馆，让人目不暇接。

既然是500销量的主编，找的茶楼自然不会很大，一个名为‘潇湘茶楼’的小门面，差点让人看不见。

“蒋主编，您好，我是赵正。”

问了前台的中年老板，赵正来到里面的8号桌，见到一位络腮胡的中年人，主动用了一个寻常的文化人称呼。

若不是提前联系过，就冲对方这相貌，赵正觉得这蒋老板年轻时肯定在社团里大杀四方过。

想必，对方不会喜欢‘老板’这个词，更中意‘主编’这个文化属性浓厚一点的情怀。
""",
# ----
"""
“赵作家，您好您好，请坐。老吴，来份套餐。”

看着这小年轻文质彬彬的模样，蒋有得热情地邀请对方坐下，顺便喊了一份早茶套餐。

虽说最近报社快撑不下去了，但是他装修公司入账不少，和文化人谈合同，必须把面子做足了。

就冲这个相貌，一看就是有知识的文化人。

难怪，负责印刷的小王觉得对方寄来的手稿不错。

只不过，一个负责印刷的小王懂个球，作为主编兼老板的蒋有得，清楚地知道这个报社近年来的市场风向。

没有什么，比得上风月小作文来得好，提升销量那叫哗哗哗的。

可惜了，去年那位胡作家因为写太嗨了，大半夜去找楼凤，结果遇到两家火拼的社团，嘎掉了。

若不然，他们小风报去年日销量就能攀升到三千，和旁边的几家老牌报社掰掰腕子。

“蒋主编，这是我按照您的要求，写的一个小说开头。”

坐下来喝了口冻柠茶，赵正干脆利落地拿出一份4000字的开头。

虽说他的存稿已经有8000，但防人之心不可无，赵正还是准备留一手。

实在不行，只能去别家碰碰运气。

毕竟，放下身段去写风月文，哪家报社都不愁稿费，无非是多是少。

“好，赵作家吃点早茶，我先看看。”

看到对方如此干脆，蒋有得暗赞一声‘文化人’，继而低头看起了手稿。

作为十几年前的合盛社团双花红棍，蒋有得被砍伤双手后，退下来开了装修公司，吃了没文化的亏，少赚了不少钱。

他一直来的遗憾，就是当初在小学的时候没有好好学习，长大后只能打打杀杀，半辈子也就混个五百来万的身家。

去年买下报社之后，倒亏50万，蒋有得对文化人的需求就更迫切了。

“写得好。”

前面的铺垫情节一扫而过，蒋有得看到那两千字的风月情节，内心里有一种回家找老婆打扑克的冲动，忍不住拍案叫好。

作为去年‘转职’的报社主编，蒋有得也算是遍览群书，自然能看得出一本小说的好坏。

像其它报社的小颜色文，和各大影院深夜里放的二级以下片子没啥区别，看多了就腻。

这位赵作家写的圣女，杀伐果断如将军，绝色倾城像女神，却是被阴险狡诈的魔师拿下，那叫一个酣畅...的爽。

对，就是看得爽，看得有劲。

单凭这个开头，就能让他们报纸销量上涨个两三百。

“不知道蒋主编觉得如何？”

等对方放下手稿，赵正也放下了手中的玻璃杯，开口问道。

“写得很不错，我可以给出千字50的价格，不知道赵作家觉得怎么样？”

满意地点点头，蒋有得说出了自己的报价。

他觉得，这本小说有火的潜质。

按照他们小风报这样的报纸，二分之一的篇幅都是小说，剩下的就是各类借鉴来的花边新闻和社论。

曾经，他们那本《七侠风月令》连载的时候，还一天一期，每期三千字小说，照样天天销量过千，那也是小风报最辉煌的时候，给出的稿费价格自然高了去了。

现在嘛，为了节省成本，他们大概是两天一期，一期八版，四版差不多5000字的小说。

最近一段时间，因为没有什么出色的小说，每天也就两百份不到，90份出头，都是社团里的报刊点卖出去的。

若是有这本小说，一期卖出个1000份肯定没问题。

按照一期卖个1000份，1份1.5港元，报社总收入700港元，加上人工成本，小说这边的开支控制在千字50元左右。

这么算下来，两天发表5000字，文字成本也要250块钱，算是报纸成本的大头了。

最主要的是，文化人写书费劲，他也不可能做到两天就有5000字的小说发表，等稿期间只能拿别的垃圾书凑数。

这年头，一本风月小说为主的小报纸，全靠小说作家养着，没得办法。

“蒋主编，我可以保证，每天给你们5000字的小说。”

对于这个价格，赵正没有予以置评，而是说出了自己的实力。

千字50元看似不少，比其余几家报社给的报价都高不少，但按照这样的价格，一个月顶多三千多块，完全不符合赵正的预期。

昨天一个晚上，他写顺手了都有8000字的手稿，每天拿出5000字，并不困难，抽空还能和小犹太出门散散心。

没办法，整本书都在他脑海里，写出来全都是钱...都是文学。

一天250起步，一个月也有个七千五了，勉强算得上现在普通小白领的工资水准。

当然，他的产量若是上来了，报纸销量增长，总不能还是千字50吧。

为了爱情，他要卷，把同行都卷不行了，那他就是最值钱的那个！！！

卷死同行，就地称王！

“一天5000字？”

听到对方的话，蒋有得忍不住瞪大了双眼。

这速度，简直完爆现在的知名作家，像什么况尼兄妹和这条街最火的‘伯光’、‘多凤’都弱爆了。

要是能有5000字一天的小说，那不是能把小报纸的潜在用户都拉过来。

只不过，惊讶和欣喜过后，蒋有得就担心起了报纸的销量。

若是报纸销量上去，一切都好说，只要能过千，蒋有得就可以给出千字80的价格，甚至更高。

想到这里，蒋有得不太确定地问道：“赵作家，每天5000字，能保持着相同的水准吗？”

怕就怕，数量上去了，这小说质量就不行了，读者不买账，一样是白瞎。

“咱们三万字一签约，如果质量不行，蒋主编也不会跟我续约吧。”

脑海里有着180万字全书情节的赵正，不慌不忙地说道。

得益于景老先生和况尼兄妹等人的偶像效应，很多不想端盘子、当保安的年轻人涌入写作这个行业，现在的非主流小报社和小说签约作家的条款很宽松。

几万字算一次稿酬，不好就结束合作，甚至还有一万字一签约的。

和现在边拍边播的电视剧一样，收视率不佳，观众不买账，直接切了，换一部剧，用最少的成本赚最多的钱。

电视剧如此，小报社也是如此，甚至名报、太阳星这样的主流报社也是如此，没有人会为错误买单。

“赵作家既然说得这么爽快了，我老蒋也不含糊。这样，前面三万字我给千字80的润笔费，后续看报纸销量，给您加。”

知道对方的意思，蒋有得主动说起了这个稿费。

至于后续加不加，那就看报纸的销量了。

“可以。”

没有讨价还价，赵正答应了这个价码。

一穷二白、没有任何名气的他，根本没有谈价钱的资本。

这年头，只要能提升报纸销量，他的稿费自然会增加。

“那咱们签个约。”

见这年轻作家如此干脆，蒋有得也是快速拿出一张准备好的制式合约。

合约上面，只有简单的甲方乙方，言明乙方将某小说的文字版权授予甲方用作报刊发行使用，稿费多少，还特地表明了后续稿费随行就市，会持续增减。

至于其余的版权，全都属于乙方作者本人。

这年头，出书的小说作家寥寥无几，大多数港城市民更喜欢报纸连载的方式，类似于多年后小说网站的日常追更。

每天的茶楼里头，都有一些闲得发慌、收租过日子的市民，坐在那里喝茶看报，顺便交流小说的看法，一起骂主角和作者。

当然，还有部分是白天上班累的白领，下班回去坐在茶楼里也能参与进去，顺便发泄一下被上司和现实打压的郁闷。

“等等。”

就在赵正准备签约的时候，一个声音在门口响起。
""",
]

def easy_unittest():
	import torch
	import logging
	from peft import PeftModel
	from transformers import AutoModelForCausalLM

	device = "cpu"
	base_model_path = "/nfsshare/home/caoyang/resource/model/Qwen/Qwen3-8B-Instruct"
	output_dir_1 = "/nfsshare/home/caoyang/caoyang/easyllm/temp/sft-7b/sft+Qwen3-8B-Instruct+MATH-500+20250924074338"
	output_dir_2 = "/nfsshare/home/caoyang/caoyang/easyllm/temp/sft+Qwen3-8B-Instruct+gsm8k+20251007220921"
	model_1 = AutoModelForCausalLM.from_pretrained(base_model_path).to(device)
	model_1 = PeftModel.from_pretrained(model_1, output_dir_1)
	model_1 = model_1.merge_and_unload()
	model_1 = PeftModel.from_pretrained(model_1, output_dir_2) 
	model_1 = model_1.merge_and_unload()

	model_2 = AutoModelForCausalLM.from_pretrained(base_model_path).to(device)
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
