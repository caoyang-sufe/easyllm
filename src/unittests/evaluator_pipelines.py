# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@stu.sufe.edu.cn
# Evaluator for CAUSAL_LM

import os
import json
import time
import torch
import logging
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, PeftModel
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from src.unittests import (
	model_home, dataset_home, model_names, dataset_names, evaluate_home, 
	dataset_processors_map, dataset_train_test_splits_map, model_parallel_classes_map
)
from src.pipelines.evaluator import base_pipeline
from src.modules import (
	ParallelQwen2Model, SkipLayerQwen2ForCausalLM,
	ParallelQwen2ForCausalLM, SkipLayerQwen2ForCausalLM,
	ParallelQwen3Model, SkipLayerQwen3ForCausalLM,
	ParallelQwen3ForCausalLM, SkipLayerQwen3ForCausalLM,
	ParallelLlamaModel, SkipLayerLlamaForCausalLM,
	ParallelLlamaForCausalLM, SkipLayerLlamaForCausalLM,
	SkipLayerDeepseekModel, SkipLayerDeepseekForCausalLM,
	ParallelDeepseekModel, ParallelDeepseekForCausalLM,
	SkipLayerDeepseekV2Model, SkipLayerDeepseekV2ForCausalLM,
	ParallelDeepseekV2Model, ParallelDeepseekV2ForCausalLM,
	SkipLayerDeepseekV3Model, SkipLayerDeepseekV3ForCausalLM,
	ParallelDeepseekV3Model, ParallelDeepseekV3ForCausalLM,
)

def easy_evaluate(
	model_id = 10, 
	dataset_ids = [7, 5],
	adapter_output_dirs = None,
	n_cuda=2, 
	do_sample=False, 
	adapter_output_dirs=None,
):
	model_name_or_path = os.path.join(model_home, model_names[model_id])
	logging.info(f"Load model: {model_name_or_path} ...")
	tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
	if overwritten_model_class is None:
		logging.info("  - Using AutoModelForCausalLM ...")
		device = "cuda" if torch.cuda.is_available() else "cpu"
		model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto")
	else:
		logging.info(f"  - Using {overwritten_model_class} ...")
		device = "cuda:0"
		model = eval(overwritten_model_class).from_pretrained(model_name_or_path, n_cuda = n_cuda)
		model.module_to_device()
	if adapter_output_dirs is not None:
		logging.info(f"  - Load adapters ...")
		for i, adapter_output_dir in enumerate(adapter_output_dirs):
			logging.info(f"    - Load adapter {i}: {adapter_output_dir}")
			model = PeftModel.from_pretrained(model, model_id = adapter_output_dir)
			model = model.merge_and_unload()

def evaluate_math_500(model_id=10, overwritten_model_class=None, n_cuda=2, do_sample=False, adapter_output_dirs=None):
	model_name_or_path = os.path.join(model_home, model_names[model_id])
	logging.info(f"Load model: {model_name_or_path} ...")
	tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
	if overwritten_model_class is None:
		logging.info("  - Using AutoModelForCausalLM ...")
		device = "cuda" if torch.cuda.is_available() else "cpu"
		model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto")
	else:
		logging.info(f"  - Using {overwritten_model_class} ...")
		device = "cuda:0"
		model = eval(overwritten_model_class).from_pretrained(model_name_or_path, n_cuda = n_cuda)
		model.module_to_device()
	if adapter_output_dirs is not None:
		logging.info(f"  - Load adapters ...")
		for i, adapter_output_dir in enumerate(adapter_output_dirs):
			logging.info(f"    - Load adapter {i}: {adapter_output_dir}")
			model = PeftModel.from_pretrained(model, model_id = adapter_output_dir)
			model = model.merge_and_unload()
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
		"do_sample": do_sample,
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
	time_string = time.strftime("%Y%m%d%H%M%S")
	if do_sample:
		logging.info("Sampling Evaluation ...")
		metric_summary = base_pipeline(**kwargs)
		model_name = model_name_or_path.split('/')[-1]
		with open(f"./temp/{model_name}+{dataset_path.split('/')[-1]}+dosample+{time_string}.json", 'w', encoding="utf8") as f:
			json.dump(metric_summary, f, ensure_ascii=False)
	else:
		logging.info("Greedy Evaluation...")
		metric_summary = base_pipeline(**kwargs)
		model_name = model_name_or_path.split('/')[-1]
		with open(f"./temp/{model_name}+{dataset_path.split('/')[-1]}+greedy+{time_string}.json", 'w', encoding="utf8") as f:
			json.dump({**{"adapter_output_dirs": adapter_output_dirs}, **metric_summary}, f, ensure_ascii=False)

def evaluate_gsm8k(model_id=10, overwritten_model_class=None, n_cuda=2, do_sample=False, adapter_output_dirs=None):
	model_name_or_path = os.path.join(model_home, model_names[model_id])
	logging.info(f"Load model: {model_name_or_path} ...")
	tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
	if overwritten_model_class is None:
		logging.info("  - Using AutoModelForCausalLM ...")
		device = "cuda" if torch.cuda.is_available() else "cpu"
		model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto")
	else:
		logging.info(f"  - Using {overwritten_model_class} ...")
		device = "cuda:0"
		model = eval(overwritten_model_class).from_pretrained(model_name_or_path, n_cuda = n_cuda)
		model.module_to_device()
	if adapter_output_dirs is not None:
		logging.info(f"  - Load adapters ...")
		for i, adapter_output_dir in enumerate(adapter_output_dirs):
			logging.info(f"    - Load adapter {i}: {adapter_output_dir}")
			model = PeftModel.from_pretrained(model, model_id = adapter_output_dir)
			model = model.merge_and_unload()
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
		"do_sample": do_sample,
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
	time_string = time.strftime("%Y%m%d%H%M%S")
	if do_sample:
		logging.info("Sampling Evaluation ...")
		metric_summary = base_pipeline(**kwargs)
		model_name = model_name_or_path.split('/')[-1]
		with open(f"./temp/{model_name}+{dataset_path.split('/')[-1]}+dosample+{time_string}.json", 'w', encoding="utf8") as f:
			json.dump({**{"adapter_output_dirs": adapter_output_dirs}, **metric_summary}, f, ensure_ascii=False)
	else:
		logging.info("Greedy Evaluation...")
		metric_summary = base_pipeline(**kwargs)
		model_name = model_name_or_path.split('/')[-1]
		with open(f"./temp/{model_name}+{dataset_path.split('/')[-1]}+greedy+{time_string}.json", 'w', encoding="utf8") as f:
			json.dump({**{"adapter_output_dirs": adapter_output_dirs}, **metric_summary}, f, ensure_ascii=False)

def evaluate_leetcodedataset(model_id=10, overwritten_model_class=None, n_cuda=2, do_sample=False, adapter_output_dirs=None):
	model_name_or_path = os.path.join(model_home, model_names[model_id])
	logging.info(f"Load model: {model_name_or_path} ...")
	tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
	if overwritten_model_class is None:
		logging.info("  - Using AutoModelForCausalLM ...")
		device = "cuda" if torch.cuda.is_available() else "cpu"
		model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto")
	else:
		logging.info(f"  - Using {overwritten_model_class} ...")
		device = "cuda:0"
		model = eval(overwritten_model_class).from_pretrained(model_name_or_path, n_cuda = n_cuda)
		model.module_to_device()
	if adapter_output_dirs is not None:
		logging.info(f"  - Load adapters ...")
		for i, adapter_output_dir in enumerate(adapter_output_dirs):
			logging.info(f"    - Load adapter {i}: {adapter_output_dir}")
			model = PeftModel.from_pretrained(model, model_id = adapter_output_dir)
			model = model.merge_and_unload()
	dataset_path = os.path.join(dataset_home, dataset_names[6])
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
		"input_column": "query",
		"target_column": "response",
		"do_sample": do_sample,
		"do_sample_times": 10,
		"do_sample_kwargs": {"top_k": 0, "top_p": 1., "temperature": 1., "num_beams": 1},
		"use_cache": True,
		"input_max_length": 1024,
		"target_max_length": 512,
		"metrics": [
			("calc_token_accuracy", {}, "token_accuracy"),
			("calc_perplexity", {}, "perplexity"),
			("calc_bleu", {"min_grams": 1, "max_grams": 3}, "bleu_3"),
			("calc_rouge_n", {'n': 3, "beta": 1}, "rouge_3"),
			("calc_rouge_w", {"weight_function": lambda _x: _x, "weight_function_reverse": lambda _x: _x, "beta": 1}, "rouge_l"),
			("calc_rouge_w", {"weight_function": lambda _x: _x ** 2, "weight_function_reverse": lambda _x: _x ** 0.5, "beta": 1}, "rouge_w"),
		],
	}
	time_string = time.strftime("%Y%m%d%H%M%S")
	if do_sample:
		logging.info("Sampling Evaluation ...")
		metric_summary = base_pipeline(**kwargs)
		model_name = model_name_or_path.split('/')[-1]
		with open(f"./temp/{model_name}+{dataset_path.split('/')[-1]}+dosample+{time_string}.json", 'w', encoding="utf8") as f:
			json.dump({**{"adapter_output_dirs": adapter_output_dirs}, **metric_summary}, f, ensure_ascii=False)
	else:
		logging.info("Greedy Evaluation...")
		metric_summary = base_pipeline(**kwargs)
		model_name = model_name_or_path.split('/')[-1]
		with open(f"./temp/{model_name}+{dataset_path.split('/')[-1]}+greedy+{time_string}.json", 'w', encoding="utf8") as f:
			json.dump({**{"adapter_output_dirs": adapter_output_dirs}, **metric_summary}, f, ensure_ascii=False)

def evaluate_chinese_poems(model_id=10, overwritten_model_class=None, n_cuda=2, do_sample=False, adapter_output_dirs=None):
	model_name_or_path = os.path.join(model_home, model_names[model_id])
	logging.info(f"Load model: {model_name_or_path} ...")
	tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
	if overwritten_model_class is None:
		logging.info("  - Using AutoModelForCausalLM ...")
		device = "cuda" if torch.cuda.is_available() else "cpu"
		model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto")
	else:
		logging.info(f"  - Using {overwritten_model_class} ...")
		device = "cuda:0"
		model = eval(overwritten_model_class).from_pretrained(model_name_or_path, n_cuda = n_cuda)
		model.module_to_device()
	if adapter_output_dirs is not None:
		logging.info(f"  - Load adapters ...")
		for i, adapter_output_dir in enumerate(adapter_output_dirs):
			logging.info(f"    - Load adapter {i}: {adapter_output_dir}")
			model = PeftModel.from_pretrained(model, model_id = adapter_output_dir)
			model = model.merge_and_unload()
	dataset_path = os.path.join(dataset_home, dataset_names[7])
	dataset = load_dataset(dataset_path, split="train[1000:1100]")
	data_processor = lambda _data: {"prompt": _data["content"][:-10], "completion": _data["response"][-10:]}
	dataset = dataset.map(data_processor)
	logging.info(f"Use dataset: {dataset_path} ...")
	kwargs = {
		"model": model,
		"tokenizer": tokenizer,
		"dataset": dataset,
		"model_name_or_path": None,
		"dataset_name_or_path": dataset_path,
		"test_data_split": None,
		"test_data_size": None,
		"device": device,
		"input_column": "query",
		"target_column": "response",
		"do_sample": do_sample,
		"do_sample_times": 10,
		"do_sample_kwargs": {"top_k": 0, "top_p": 1., "temperature": 1., "num_beams": 1},
		"use_cache": True,
		"input_max_length": 256,
		"target_max_length": 32,
		"metrics": [
			("calc_token_accuracy", {}, "token_accuracy"),
			("calc_perplexity", {}, "perplexity"),
			("calc_bleu", {"min_grams": 1, "max_grams": 3}, "bleu_3"),
			("calc_rouge_n", {'n': 3, "beta": 1}, "rouge_3"),
			("calc_rouge_w", {"weight_function": lambda _x: _x, "weight_function_reverse": lambda _x: _x, "beta": 1}, "rouge_l"),
			("calc_rouge_w", {"weight_function": lambda _x: _x ** 2, "weight_function_reverse": lambda _x: _x ** 0.5, "beta": 1}, "rouge_w"),
		],
	}
	time_string = time.strftime("%Y%m%d%H%M%S")
	if do_sample:
		logging.info("Sampling Evaluation ...")
		metric_summary = base_pipeline(**kwargs)
		model_name = model_name_or_path.split('/')[-1]
		with open(f"./temp/{model_name}+{dataset_path.split('/')[-1]}+dosample+{time_string}.json", 'w', encoding="utf8") as f:
			json.dump({**{"adapter_output_dirs": adapter_output_dirs}, **metric_summary}, f, ensure_ascii=False)
	else:
		logging.info("Greedy Evaluation...")
		metric_summary = base_pipeline(**kwargs)
		model_name = model_name_or_path.split('/')[-1]
		with open(f"./temp/{model_name}+{dataset_path.split('/')[-1]}+greedy+{time_string}.json", 'w', encoding="utf8") as f:
			json.dump({**{"adapter_output_dirs": adapter_output_dirs}, **metric_summary}, f, ensure_ascii=False)
