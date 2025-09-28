# -*- coding: utf-8 -*-
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
