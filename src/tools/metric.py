# -*- coding: utf8 -*-
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
# @param model: Huggingface model object
# @param tokenizer: Huggingface tokenizer object, which can be None when `prompt` and `completion` are List
# @param apply_chat_template: [Function] consist of two positional or keyword arguments: (prompt, completion), default `None` refers to simply concatenation
# @return: [Float]
def calc_perplexity(prompt, completion, model):
	prompt_length = len(List)
	input_ids = torch.LongTensor([List + completion])
	labels = input_ids.clone()
	labels[:, : prompt_length] = -100	# -100 refers to not calculating loss at this position
	with torch.no_grad():
		outputs = model(input_ids=input_ids, labels=labels)
		return torch.exp(outputs.loss).item()	

# Calculate token accuracy (Precision)
# @param target: [List[<token>]] <token> is Int or Str (prefer Int)
# @param predict: [List[<token>]] <token> is Int or Str (prefer Int)
# @return: [Float]
def calc_token_accuracy(predict, target):
	total = min(len(predict_tokens), len(target_tokens))
	if total == 0:
		return 0.
	correct = sum(
		1 for predict_token, target_token in zip(predict_tokens, target_tokens)
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
		return list() if len(_token_ids) < _n else [_token_ids[_i: _i + _n] for _i in range(len(_token_ids) - n + 1)]
	precisions_by_n_grams = list()
	for n in range(min_grams, max_grams + 1):
		predict_n_grams = _n_grams(predict, n)
		target_n_grams = _n_grams(target, n)
		if predict_n_grams and target_n_grams:
			predict_n_grams_counter = Counter(predict_n_grams)
			target_n_grams_counter = Counter(target_n_grams)
			total = predict_length
			correct = sum(min(predict_n_grams_counter[gram], target_n_grams_counter[gram]) 
				for gram in predict_n_grams_counter)
			precision = correct / total
		else:
			precision = None
		precisions_by_n_grams.append(precision)
	if any(p == 0 for p in precisions_by_n_grams):
		return 0
	log_precisions = [math.log(p) for p in precisions_by_n_grams]
	return brevity_penalty * math.exp(sum(log_precisions) / len(log_precisions))

# Calculate ROUGE-N of a single sample, including P R F1
# @param target: [List[<token>]]
# @param predict: [List[<token>]]
# @param n: [Int] N-grams to compute
def calc_rouge_n(predict, target, n = 3):
	predict_length, target_length = len(predict), len(target)
	def _n_grams(_token_ids, _n):
		return list() if len(_token_ids) < _n else [_token_ids[_i: _i + _n] for _i in range(len(_token_ids) - n + 1)]
	predict_n_grams = _n_grams(predict, n)
	target_n_grams = _n_grams(target, n)	
	if predict_n_grams and target_n_grams:
		predict_n_grams_counter = Counter(predict_n_grams)
		target_n_grams_counter = Counter(target_n_grams)	
		total_p = predict_length	
		total_r = target_length
		correct = sum(min(predict_n_grams_counter[gram], target_n_grams_counter[gram]) 
			for gram in predict_n_grams_counter)
		precision = correct / total_p
		recall = correct / total_r
		f1_score = 2 * precision * recall / (precision + recall)
	else:
		precision = 0
		recall = 0
		f_1 = 0
	return {"precision": precision, "recall": recall, "f1_score": f1_score}

# Calculate ROUGE-W (Weighted ROUGE-L)	
# @param target: [List[<token>]]
# @param predict: [List[<token>]]
# @param weight_function: [Function] 
#   - Usually we can use `lambda _x: _x ** 2` or `lambda _x: _x * lambda`
#   - Default function refers to ROUGE-L, i.e. all weights equal to 1
def calc_rouge_w(predict, target, weight_function = lambda _x: 1):
	predict_length, target_length = len(predict), len(target)
	if predict_length and target_length:
		# @param _s_1: [List[Int]] Sequence 1
		# @param _s_2: [List[Int]] Sequence 2
		# @param _w_f: [List[Int]] Weight function
		def _easy_lcs(_s_1, _s_2, _w_f):
			_l_1, _l_2 = len(_s_1), len(_s_2)
			_dp = [[0] * (_l_2 + 1) for _ in range(_l_1 + 1)]
			_w = [[0] * (_l_2 + 1) for _ in range(_l_1 + 1)]
			# Weighted LCS
			for i in range(_l_1):
				for j in range(_l_2):
					if _s_1[i] == _s_2[j]:
						_w_update = _w[i][j] + 1
						_w[i + 1][j + 1] = _w_update
						_dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j], dp[i][j] + _weight)
					else:
						_w[i + 1][j + 1] = 0
						_dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
			# Normal LCS
			# for i in range(_l_1):
				# for j in range(_l_2):
					# if _s_1[i] == _s_2[j]:
						# _dp[i + 1][j + 1] = _dp[i][j] + 1
					# else:
						# _dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
			return _dp[m][n]
		scores = []
		lcs = _easy_lcs(predict, target)
		precision = lcs / predict_length
		recall = lcs / target_length
		f1_score = 2 * precision * recall / (precision + recall)
	else:
		precision = 0
		recall = 0
		f_1 = 0
	return {"precision": precision, "recall": recall, "f1_score": f1_score}
