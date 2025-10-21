# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@stu.sufe.edu.cn
# Token-in-token-out Metrics
# Prefer each function with two keyword arguments: (predict: List[Int], target: List[Int])

import os
import math
import types
import torch
import numpy
import logging
import evaluate
from torch.nn import functional as F
from collections import Counter

# Calculate perplexity of a single sample, loss on completion only
# @param prompt: [List[<token_id>]]
# @param completion: [List[<token_id>]]
# @param model: Huggingface AutoModelForCausalLM object
# @param device: [Str|torch.device]
# @return: [Float]
def calc_perplexity(prompt, completion, model, device="cuda"):
	prompt_length = len(prompt)
	input_ids = torch.LongTensor([prompt + completion]).to(device)
	labels = input_ids.clone().to(device)
	labels[:, :prompt_length] = -100	# -100 refers to not calculating loss at this position
	with torch.no_grad():
		outputs = model(input_ids=input_ids, labels=labels)
		return torch.exp(outputs.loss).item()

# Calculate token accuracy (Precision)
# @param target: [List[<token>]] <token> is Int or Str (prefer Int)
# @param predict: [List[<token>]] <token> is Int or Str (prefer Int)
# @return: [Float]
def calc_token_accuracy(predict, target):
	total = min(len(predict), len(target))
	if total == 0:
		return 0.
	correct = sum(
		1 for predict_token, target_token in zip(predict, target)
		if predict_token == target_token
	)
	return correct / total

# Calculate BLEU of a single sample
# @param target: [List[<token>]]
# @param predict: [List[<token>]]
# @param max_grams: [Int] Maximum grams to compute
# @return: [Float]
def calc_bleu(predict, target, min_grams = 1, max_grams = 3):
	predict_length, target_length = len(predict), len(target)
	brevity_penalty = 1 if predict_length > target_length else math.exp(1 - predict_length / target_length)
	def _n_grams(_token_ids, _n):
		return list() if len(_token_ids) < _n else [tuple(_token_ids[_i: _i + _n]) for _i in range(len(_token_ids) - n + 1)]
	precisions_by_n_grams = list()
	for n in range(min_grams, max_grams + 1):
		predict_n_grams = _n_grams(predict, n)
		target_n_grams = _n_grams(target, n)
		if predict_n_grams and target_n_grams:
			predict_n_grams_counter = Counter(predict_n_grams)
			target_n_grams_counter = Counter(target_n_grams)
			total = len(predict_n_grams)
			correct = sum(min(predict_n_grams_counter[gram], target_n_grams_counter[gram])
				for gram in predict_n_grams_counter)
			precision = correct / total
			precisions_by_n_grams.append(precision)
	if (not precisions_by_n_grams) or any(p == 0 for p in precisions_by_n_grams):
		return 0
	log_precisions = [math.log(p) for p in precisions_by_n_grams]
	return brevity_penalty * math.exp(sum(log_precisions) / len(log_precisions))

# Calculate ROUGE-N of a single sample, including P R F1
# @param target: [List[<token>]]
# @param predict: [List[<token>]]
# @param n: [Int] N-grams to compute
# @param beta: [Float] F1 weight
# @return: Dict["prediction": Float, "recall": Float, "f1_score": Float]
def calc_rouge_n(predict, target, n = 3, beta = 1):
	predict_length, target_length = len(predict), len(target)
	def _n_grams(_token_ids, _n):
		return list() if len(_token_ids) < _n else [tuple(_token_ids[_i: _i + _n]) for _i in range(len(_token_ids) - n + 1)]
	predict_n_grams = _n_grams(predict, n)
	target_n_grams = _n_grams(target, n)
	if predict_n_grams and target_n_grams:
		predict_n_grams_counter = Counter(predict_n_grams)
		target_n_grams_counter = Counter(target_n_grams)
		total_p = len(predict_n_grams)
		total_r = len(target_n_grams)
		correct = sum(min(predict_n_grams_counter[gram], target_n_grams_counter[gram])
			for gram in predict_n_grams_counter)
		precision = correct / total_p
		recall = correct / total_r
		denominator = (beta ** 2) * precision + recall
		f1_score = 0 if denominator == 0 else (1 + beta ** 2) * precision * recall / denominator
	else:
		precision = 0
		recall = 0
		f1_score = 0
	return {"precision": precision, "recall": recall, "f1_score": f1_score}

# Calculate ROUGE-W (Weighted ROUGE-L): https://aclanthology.org/W04-1013.pdf
# @param target: [List[<token>]]
# @param predict: [List[<token>]]
# @param weight_function: [Function]
#   - Default function refers to the implementation in  https://aclanthology.org/W04-1013.pdf for more details
#   - Set `weight_function = lambda _x: _x` then degrade to ROUGE-L
# @param weight_function_reverse: [Function] the reverse function of weight_function
# @param beta: [Float] F1 weight
# @return: Dict["prediction": Float, "recall": Float, "f1_score": Float]
def calc_rouge_w(predict,
				 target,
				 weight_function = lambda _x: _x,
				 weight_function_reverse = lambda _x: _x,
				 beta = 1
				 ):
	predict_length, target_length = len(predict), len(target)
	if predict_length and target_length:
		# @param _x: [List[Int]] Sequence 1
		# @param _y: [List[Int]] Sequence 2
		# @param _f: [List[Int]] Weight function
		def _weighted_lcs(_x, _y, _f):
			_m, _n = len(_x), len(_y)
			_c = [[0] * (_n + 1) for _ in range(_m + 1)]
			_w = [[0] * (_n + 1) for _ in range(_m + 1)]
			# Weighted LCS
			for _i in range(_m):
				for _j in range(_n):
					if _x[_i] == _y[_j]:
						_k = _w[_i][_j]
						_c[_i + 1][_j + 1] = _c[_i][_j] + _f(_k + 1) - _f(_k)
						_w[_i + 1][_j + 1] = _k + 1
					else:
						_c[_i + 1][_j + 1] = max(_c[_i][_j + 1], _c[_i + 1][_j])
						_w[_i + 1][_j + 1] = 0
			return _c[_m][_n]
		weighted_lcs = _weighted_lcs(predict, target, weight_function)
		precision = weight_function_reverse(weighted_lcs / weight_function(predict_length))
		recall = weight_function_reverse(weighted_lcs / weight_function(target_length))
		denominator = (beta ** 2) * precision + recall
		f1_score = 0 if denominator == 0 else (1 + beta ** 2) * precision * recall / denominator
	else:
		precision = 0
		recall = 0
		f1_score = 0
	return {"precision": precision, "recall": recall, "f1_score": f1_score}


# ----------------------------------------------------------------------
# Generate keyword arguments `compute_metrics` for `transformers.Trainer.__init__`
# @param metrics: [List[Str] | List[Tuple(Str, Dict, Str)]]
#   - e.g. ["bleu", "rouge"] or [evaluate.load("rouge"), evaluate.load("bleu")] when `strategy = "evaluate"`
#   - e.g. [("calc_token_accuracy", {}, "token_accuracy"), ("calc_rouge_n", {'n': 3, "beta": 1}, "rouge_3")] when `strategy = 'diy'`
# @param strategy: [Str] e.g. "evaluate" or "diy", which accept different formats of keyword arguments
# @param evaluate_home: [Str] e.g. default import from src.unittests, which is the github repository of "https://github.com/huggingface/evaluate"
# @param tokenizer: Huggingface tokenizer object
# @return _compute_metrics: [Function]
def generate_compute_metrics_function(metrics = ["bleu", "rouge"],
									  strategy = "evaluate",
									  evaluate_home = None,
									  tokenizer = None,
									  ):
	assert isinstance(metrics, list), f"Expect List but got {metrics}"
	if strategy == "evaluate":
		# May encounter network error
		assert tokenizer is not None, f"Need tokenizer in {strategy} mode"
		if hasattr(metrics[0], "__module__") and metrics[0].__module__.startswith("evaluate_modules.metrics"):
			logging.info("Metrics from callable functions!")
			metric_name_to_function = {metric.name: metric for metric in metrics}
		elif isinstance(metrics[0], str):
			logging.info("Load metrics ...")
			from datasets import DownloadConfig
			download_config = DownloadConfig(use_etag=False)
			if evaluate_home is None:
				from src.unittests import evaluate_home
			metric_name_to_function = dict()
			for metric_name in metrics:
				metric_path = os.path.join(evaluate_home, "metrics", metric_name, f"{metric_name}.py")
				logging.info(f"  - Load {metric_name} from {metric_path}")
				metric_function = evaluate.load(metric_path, download_config=download_config)
				logging.info(f"    + ok!")
				metric_name_to_function[metric_name] = metric_function
		else:
			raise Exception(f"Unknown metric type (expect Str or Function): {type(metrics[0])}")
		# Get global `metric_name_to_function` like {"bleu": evaluate.load("bleu"), ...}
		# @param _eval_prediction: [transformers.trainer_utils.EvalPrediction]
		def _compute_metrics(_eval_prediction, **_kwargs):
			_predictions, _label_ids = _eval_prediction.predictions, _eval_prediction.label_ids	# Float(batch_size, seq_length, n_vocab), Long(batch_size, seq_length)
			_prediction_ids = _predictions.argmax(-1)	# Float(batch_size, seq_length, n_vocab) => Long(batch_size, seq_length)
			_prediction_ids_to_list, _label_ids_to_list = list(), list()
			_prediction_ids = [_array_1[_array_2 != -100].tolist() for (_array_1, _array_2) in zip(_prediction_ids, _label_ids)] 	# Filter -100
			_label_ids = [_array[_array != -100].tolist() for _array in _label_ids] 	# Filter -100
			_decoded_predictions = tokenizer.batch_decode(_prediction_ids, skip_special_tokens=True)
			_decoded_label_ids = tokenizer.batch_decode(_label_ids, skip_special_tokens=True)
			_metric_summary = dict()
			for _metric_name, _metric_function in metric_name_to_function.items():
				_metric_summary.update(_metric_function.compute(predictions=_decoded_predictions, references=_decoded_label_ids))
			return _metric_summary
	elif strategy == "diy":
		# Token-in-token-out
		def _compute_metrics(_eval_prediction, **_kwargs):
			_batch_predictions, _batch_label_ids = _eval_prediction.predictions, _eval_prediction.label_ids	# Float(batch_size, seq_length, n_vocab), Long(batch_size, seq_length)
			_batch_prediction_ids = _batch_predictions.argmax(-1)	# Float(batch_size, seq_length, n_vocab) => Long(batch_size, seq_length)
			_metric_summary = dict()
			for _batch_prediction_id, _batch_label_id in zip(_batch_prediction_ids, _batch_label_ids):
				_batch_prediction_id = _batch_prediction_id[_batch_label_id != -100] 	# Filter predictions according to labels != -100
				_batch_label_id = _batch_label_id[_batch_label_id != -100] 	# Filter labels according to labels != -100
				assert len(_batch_prediction_id) == len(_batch_label_id), f"{len(_batch_prediction_id)} v.s. {len(_batch_label_id)}"
				for _metric_function_name, _metric_function_kwargs, _metric_name in metrics:
					_metric_value = eval(_metric_function_name)(predict=_batch_prediction_id, target=_batch_label_id, **_metric_function_kwargs)
					if _metric_function_name in ["calc_token_accuracy", "calc_bleu"]:
						# Return single value
						if not _metric_name in _metric_summary:
							_metric_summary[_metric_name] = list()
						_metric_summary[_metric_name].append(_metric_value)
					elif _metric_function_name in ["calc_rouge_n", "calc_rouge_w"]:
						# Return multiple values: P R F1
						if not _metric_name in _metric_summary:
							_metric_summary[_metric_name] = list()
						_precision, _recall, _f1_score = _metric_value["precision"], _metric_value["recall"], _metric_value["f1_score"]
						_metric_summary[_metric_name].append(_f1_score)	# len(predict) == len(target) => P == R == F1
			for _metric_name in _metric_summary:
				_metric_summary[_metric_name] = numpy.nanmean(_metric_summary[_metric_name])
			return _metric_summary
	else:
		raise Exception(f"Unexpected keyword argument: {strategy}")
	return _compute_metrics
