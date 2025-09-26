# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@stu.sufe.edu.cn
# Evaluator for CAUSAL_LM

import os
import json
import time
import logging
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from src.unittests import model_home, dataset_home, model_names, dataset_names
from src.pipelines.evaluator import base_pipeline
from src.module import (
	ParallelQwen2Model, SkipLayerQwen2ForCausalLM, 
	ParallelQwen2ForCausalLM, SkipLayerQwen2ForCausalLM, 
	ParallelQwen3Model, SkipLayerQwen3ForCausalLM, 
	ParallelQwen3ForCausalLM, SkipLayerQwen3ForCausalLM, 
	ParallelLlamaModel, SkipLayerLlamaForCausalLM, 
	ParallelLlamaForCausalLM, SkipLayerLlamaForCausalLM, 
)

def evaluate_math_500(model_id=10, parallel_model_class=None, n_cuda=2):
	model_name_or_path = os.path.join(model_home, model_names[model_id])
	logging.info(f"Load model: {model_name_or_path} ...")
	tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
	if parallel_model_class is None:
		logging.info("  - Using AutoModelForCausalLM ...")
		device = "cuda"
		model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)
	else:
		logging.info(f"  - Using {parallel_model_class} ...")
		device = "cuda:0"
		model = eval(parallel_model_class).from_pretrained(model_name_or_path, n_cuda = n_cuda)
		model.module_to_device()
	dataset_path = os.path.join(dataset_home, dataset_names[5])
	logging.info(f"Use dataset: {dataset_path} ...")
	kwargs = {
		"model": model,
		"tokenizer": tokenizer,
		"dataset": None,
		"model_name_or_path": None,
		"dataset_name_or_path": dataset_path,
		"test_data_split": "test",
		"test_data_size": 500,
		"device": device,
		"input_column": "problem",
		"target_column": "answer",
		"do_sample": False,
		"do_sample_times": 10,
		"do_sample_kwargs": {"top_k": 0, "top_p": 1., "temperature": 1., "num_beams": 1},
		"use_cache": True,
		"input_max_length": 512,
		"target_max_length": 128,
		"metrics": [
			("calc_token_accuracy", {}, "token_accuracy"),
			("calc_perplexity", {}, "perplexity"),
			("calc_bleu", {"min_grams": 1, "max_grams": 3}, "bleu_3"),
			("calc_rouge_n", {'n': 3, "beta": 1}, "rouge_3"),
			("calc_rouge_w", {"weight_function": lambda _x: _x, "weight_function_reverse": lambda _x: _x, "beta": 1}, "rouge_l"),
			("calc_rouge_w", {"weight_function": lambda _x: _x ** 2, "weight_function_reverse": lambda _x: _x ** 0.5, "beta": 1}, "rouge_w"),
		],
	}
	logging.info("Greedy Evaluation...")
	metric_summary = base_pipeline(**kwargs)
	with open(f"./temp/{model_name_or_path.split('/')[-1]}+{dataset_path.split('/')[-1]}+greedy.json", 'w', encoding="utf8") as f:
		json.dump(metric_summary, f, ensure_ascii=False)
	logging.info("Sampling Evaluation ...")
	kwargs["do_sample"] = True
	metric_summary = base_pipeline(**kwargs)
	with open(f"./temp/{model_name_or_path.split('/')[-1]}+{dataset_path.split('/')[-1]}+dosample.json", 'w', encoding="utf8") as f:
		json.dump(metric_summary, f, ensure_ascii=False)


def evaluate_gsm8k():
	model_name_or_path = os.path.join(model_home, model_names[model_id])
	logging.info(f"Load model: {model_name_or_path} ...")
	tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
	if parallel_model_class is None:
		logging.info("  - Using AutoModelForCausalLM ...")
		device = "cuda"
		model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)
	else:
		logging.info(f"  - Using {parallel_model_class} ...")
		device = "cuda:0"
		model = eval(parallel_model_class).from_pretrained(model_name_or_path, n_cuda = n_cuda)
		model.module_to_device()
	dataset_path = os.path.join(dataset_home, dataset_names[4])
	logging.info(f"Use dataset: {dataset_path} ...")
	kwargs = {
		"model": model,
		"tokenizer": tokenizer,
		"dataset": None,
		"model_name_or_path": None,
		"dataset_name_or_path": dataset_path,
		"test_data_split": "test",
		"test_data_size": None,
		"device": device,
		"input_column": "question",
		"target_column": "answer",
		"do_sample": False,
		"do_sample_times": 10,
		"do_sample_kwargs": {"top_k": 0, "top_p": 1., "temperature": 1., "num_beams": 1},
		"use_cache": True,
		"input_max_length": 512,
		"target_max_length": 128,
		"metrics": [
			("calc_token_accuracy", {}, "token_accuracy"),
			("calc_perplexity", {}, "perplexity"),
			("calc_bleu", {"min_grams": 1, "max_grams": 3}, "bleu_3"),
			("calc_rouge_n", {'n': 3, "beta": 1}, "rouge_3"),
			("calc_rouge_w", {"weight_function": lambda _x: _x, "weight_function_reverse": lambda _x: _x, "beta": 1}, "rouge_l"),
			("calc_rouge_w", {"weight_function": lambda _x: _x ** 2, "weight_function_reverse": lambda _x: _x ** 0.5, "beta": 1}, "rouge_w"),
		],
	}
	logging.info("Greedy Evaluation...")
	metric_summary = base_pipeline(**kwargs)
	with open(f"./temp/{model_name_or_path.split('/')[-1]}+{dataset_path.split('/')[-1]}+greedy.json", 'w', encoding="utf8") as f:
		json.dump(metric_summary, f, ensure_ascii=False)
	logging.info("Sampling Evaluation ...")
	kwargs["do_sample"] = True
	metric_summary = base_pipeline(**kwargs)
	with open(f"./temp/{model_name_or_path.split('/')[-1]}+{dataset_path.split('/')[-1]}+dosample.json", 'w', encoding="utf8") as f:
		json.dump(metric_summary, f, ensure_ascii=False)

