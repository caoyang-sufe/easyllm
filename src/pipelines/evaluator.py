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
from src.tools.metric import calc_mean_token_accuracy, calc_perplexity

# Base pipeline for evalution
# Currently support two metrics: "mean_token_accuracy" and "perplexity"
# @param model: Huggingface model object
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

	test_predicts = list()	# List[List[<token_id>]] or List[List[List[<token_id>]]]: the former is `do_sample=False` while the latter is `do_sample=True`
	test_targets = list()	# List[List[<token_id>]]: <token_id> is Int only
	token_accuracys = list()
	perplexitys = list()
	token_accuracy_history = list()
	perplexity_history = list()
	for i, data in enumerate(dataset):
		logging.info(f"Test data {i} ...")
		input_text = data[input_column]	# Str
		target_text = data[target_column]	# Str
		target_token_ids = tokenizer.encode(target_text, return_tensors=None)	# List[<token_id>]
		if do_sample:
			predict_token_ids_group = list()
			token_accuracy_group = list()
			perplexity_group = list()
			for j in range(do_sample_times):
				logging.info(f"  - Sample {j}")
				_, predict_token_prob, _ = generate_token_prob(
					model = model, 
					tokenizer = tokenizer, 
					prompt = input_text, 
					max_length = inputs["input_ids"].size(1) + target_max_length, 
					generate_kwargs = {"do_sample": True, "use_cache": use_cache, **do_sample_kwargs}, 
					device = device,
				)	# predict_token_prob: List[Tuple(Int, Str, Float)]
				predict_token_ids = [predict_token_prob[i][0] for i in range(len(predict_token_prob))]	# List[<token_id>]
				predict_token_ids_group.append(predict_token_ids)	# List[<token_id>] -> List[]
				token_accuracy = calc_mean_token_accuracy(predict=[predict_token_ids_group], target=[target_token_ids])
				perplexity = calc_perplexity(
					prompt = predict_token_ids_group,
					completion = target_token_ids,
					model = model,
					tokenizer = tokenizer,
					apply_chat_template = None,
				)
				logging.info(f"    - Token Accuracy: {token_accuracy}")
				logging.info(f"    - Perplexity: {perplexity}")
				token_accuracy_group.append(token_accuracy)
				perplexity_group.append(perplexity)
			token_accuracy_history.append(token_accuracy_group)
			perplexity_history.append(perplexity_group)
			token_accuracys.append(numpy.mean(token_accuracy_group))
			perplexitys.append(numpy.mean(perplexity_group))
		else:
			predict_text, predict_token_prob, predict_logits = generate_token_prob(
				model = model, 
				tokenizer = tokenizer, 
				prompt = input_text, 
				max_length = inputs["input_ids"].size(1) + target_max_length, 
				generate_kwargs = {"do_sample": False, "use_cache": use_cache, **do_sample_kwargs}, 
				device = device,
			)	# predict_token_prob: List[Tuple(Int, Str, Float)]
			predict_token_ids = [predict_token_prob[i][0] for i in range(len(predict_token_prob))]	# List[<token_id>]
			test_predicts.append(predict_token_ids)	# List[<token_id>] -> List[]
			token_accuracy = calc_mean_token_accuracy(predict=[predict_token_ids], target=[target_token_ids])
			perplexity = calc_perplexity(
				prompt = predict_token_ids,
				completion = target_token_ids,
				model = model,
				tokenizer = tokenizer,
				apply_chat_template = None,
			)
			logging.info(f"  - Token Accuracy: {token_accuracy}")
			logging.info(f"  - Perplexity: {perplexity}")
			token_accuracys.append(token_accuracy)
			perplexitys.append(perplexity)
	metrics = {
		"mean_token_accuracy": numpy.mean(token_accuracys),
		"perplexity_mean": numpy.mean(perplexitys),
	}
	if do_sample:
		return metrics, token_accuracy_history, perplexity_history
	else:
		return metrics, token_accuracys, perplexitys

	
	





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
    
