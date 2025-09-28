# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@stu.sufe.edu.cn
# Token-in-token-out Metrics
# Prefer each function with two keyword arguments: (predict: List[Int], target: List[Int])

import math
import torch
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
	labels[:, : prompt_length] = -100	# -100 refers to not calculating loss at this position
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
		
	if any(p == 0 for p in precisions_by_n_grams):
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
