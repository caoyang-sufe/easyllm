# -*- coding: utf8 -*-
# @author: caoyang
# @email: caoyang@stu.sufe.edu.cn
# Evaluator for CAUSAL_LM

import numpy
import torch
import logging
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.tools.transformers import generate_token_prob

# Base pipeline for evalution
# Currently support two metrics: "mean_token_accuracy" and "perplexity"
# @param model: Huggingface model object
# @param model: Huggingface tokenizer object
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
# @param metrics: [List[Str]] e.g. ["mean_token_accuracy", "perplexity"]
def base_pipeline(model = None,
				  tokenizer = None
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
				  metrics = ["mean_token_accuracy", "perplexity"],
				  ):
	if model is None:
		logging.info(f"Load model from {model_name_or_path}")
		model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
	if tokenizer is None:
		logging.info(f"Load tokenizer from {model_name_or_path}")
		tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)		
	if dataset is None:
		logging.info(f"Load dataset from {dataset_name_or_path} with split {test_data_split}")
		dataset = load_dataset(dataset_name_or_path, split=test_data_split)
	if model_name_or_path is None:
		model_name_or_path = model.config.name_or_pathssss
	logging.info(f"Evaluating {model_name_or_path} ...")
	if test_data_size is not None:
		if test_data_size < 1:
			dataset = dataset.select(range(int(self.dataset) * test_data_size))
		else:
			dataset = dataset.select(range(min(test_data_size, len(dataset))))
	logging.info(f"Dataset ({len(dataset)} samples): {dataset.cache_files if dataset_name_or_path is None else dataset_name_or_path}")

	test_predicts = list()	# List[Str] or List[List[Str]]
	test_targets = list()	# List[Str]
	perplexity_scores = []
	for i, data in enumerate(dataset):
		logging.info(f"Test data {i} ...")
		input_text = data[input_column]
		target_text = data[target_column]
		test_targets.append(target_text)
		if do_sample:
			predict_text_list = list()
			for _ in range(do_sample_times):
				predict_text, _, _ = generate_token_prob(
					model = model, 
					tokenizer = tokenizer, 
					prompt = input_text, 
					max_length = inputs["input_ids"].size(1) + target_max_length, 
					generate_kwargs = {"do_sample": do_sample, "use_cache": use_cache, **do_sample_kwargs}, 
					device = device,
				)
				predict_text_list.append(predict_text)	
				test_predicts.append(predict_text_list)		
		else:
			predict_text, _, _ = generate_token_prob(
				model = model, 
				tokenizer = tokenizer, 
				prompt = input_text, 
				max_length = inputs["input_ids"].size(1) + target_max_length, 
				generate_kwargs = {"do_sample": do_sample, "use_cache": use_cache, **do_sample_kwargs}, 
				device = device,
			)
			test_predicts.append(predict_text)
		
		gen_tokens
		
		# Mean token accuracy
		gen_tokens = self.tokenizer.encode(generated, add_special_tokens=False)
		ref_tokens = self.tokenizer.encode(reference, add_special_tokens=False)
		token_acc = self.mean_token_accuracy([gen_tokens], [ref_tokens])
		token_accuracies.append(token_acc)
		# Perplexity
		full_text = prompt + " " + reference
		ppl = self.perplexity([full_text])
		perplexity_scores.append(ppl)
		
		logging.info(f"  - Input text: {input_text}")
		logging.info(f"  - Target text: {target_text}")
		logging.info(f"  - Predict text: {predict_text}")
		logging.info(f"  - Token accuracy: {token_accuracy}")
		logging.info(f"  - Perplexity: {perplexity}")

	
	metrics = {
		"mean_token_accuracy": numpy.mean(token_accuracies),
		"perplexity_mean": numpy.mean(perplexity_scores),
	}
	return metrics, generated_responses, reference_completions

	
	


# @param output_tokens: [List[List[Int|Str]]]
# @param target_tokens: [List[List[Int|Str]]]
def mean_token_accuary(output_tokens, target_tokens):
	correct_tokens = 0
	total_tokens = 0
	
	for output_token, target_token in zip(predictions, references):
		# 对齐长度，取较短的长度
		min_len = min(len(pred), len(ref))
		if min_len == 0:
			continue       
		pred_tokens = pred[:min_len]
		ref_tokens = ref[:min_len]
		correct_tokens += sum(1 for p, r in zip(pred_tokens, ref_tokens) if p == r)
		total_tokens += min_len
	
	return correct_tokens / total_tokens if total_tokens > 0 else 0
	


class CausalLMEvaluator:
    def __init__(self, model_name, dataset_name):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )
        self.dataset = load_dataset(dataset_name)
        
    def mean_token_accuracy(self, predictions, references):
        correct_tokens = 0
        total_tokens = 0
        
        for pred, ref in zip(predictions, references):
            # 对齐长度，取较短的长度
            min_len = min(len(pred), len(ref))
            if min_len == 0:
                continue       
            pred_tokens = pred[:min_len]
            ref_tokens = ref[:min_len]
            correct_tokens += sum(1 for p, r in zip(pred_tokens, ref_tokens) if p == r)
            total_tokens += min_len
        
        return correct_tokens / total_tokens if total_tokens > 0 else 0
    
    def perplexity(self, texts):
        perplexities = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                ppl = torch.exp(loss).item()
                perplexities.append(ppl)
        return numpy.mean(perplexities)
    
