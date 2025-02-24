# -*- coding: utf8 -*-
# @author: caoyang
# @email: caoyang@stu.sufe.edu.cn

import torch
from copy import deepcopy
from torch.nn import functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM

# @param model: Huggingface model object
# @param tokenizer: Huggingface tokenizer Object
# @param input_text: Str
# @param max_length: Int, the number of tokens to be generated
# @param device: Str, e.g. "cuda" or "cpu"
# @return generated_text: Str
def greedy_decode(model,
				  tokenizer,
				  input_text, 
				  max_length,
				  device = "cuda",
				  ):
	inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)	# Str => Long(1, n_tokens)
	past_key_values = None
	for i in range(max_length):
		outputs = model(inputs, past_key_values=past_key_values)
		logits = outputs.logits	# Float(1, n_tokens + i + 1, n_vocab), where `n_vocab` is 151936 in DeepSeek-R1-Distill-Qwen
		past_key_values = outputs.past_key_values	# Dictlike[key_cache: Float(1, 2, X, hidden_size), value_cache: Float(1, 2, X, hidden_size)], where X = (i + 1) * (n_tokens + i / 2)
		next_token_id = torch.argmax(logits[:, -1, :], dim=-1)	# Float(1, n_tokens + i + 1, n_vocab) => Float(1, n_vocab) => Long(1, )
		inputs = torch.cat([inputs, next_token_id.unsqueeze(-1)], dim=-1)	# Long(1, n_tokens + i) => Long(1, n_tokens + i + 1) 
	generated_text = tokenizer.decode(inputs[0], skip_special_tokens=True)	# Long(n_tokens + 1, ) => Str 
	return generated_text

# @param model: Huggingface model object
# @param tokenizer: Huggingface tokenizer Object
# @param input_text: Str
# @param max_length: Int, the number of tokens to be generated
# @param n_branches: Int, the number of branches searched each step 
# @param depth: Int, the number of step searched
# @param device: Str, e.g. "cuda" or "cpu"
# @return generated_text: Str
def beam_search_decode(model,
					   tokenizer,
					   input_text,
					   max_length,
					   n_branches,
					   depth,
					   device = "cuda",
					   ):
	inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)	# Str => Long(1, n_tokens)
	n_tokens = inputs.size(1)
	past_key_values = None
	def _easy_beam_search(_depth, _inputs, _past_key_values, _prob):
		if _depth == 0:
			candidates.append((_inputs[0], _prob))
		else:
			_outputs = model(_inputs, past_key_values=_past_key_values)
			_logits = _outputs.logits	
			_past_key_values = _outputs.past_key_values
			_next_token_probs = F.softmax(_logits[:, -1, :], dim=-1)
			_next_topk_tokens_probs, _next_topk_tokens_ids = torch.topk(_next_token_probs, k=n_branches, dim=-1)
			for _candidate_token_prob, _candidate_token_id in zip(_next_topk_tokens_probs[0], _next_topk_tokens_ids[0]):
				_candidate_inputs = torch.cat([_inputs, _candidate_token_id.unsqueeze(-1).unsqueeze(-1)], dim=-1)
				_easy_beam_search(_depth - 1, _candidate_inputs, _past_key_values, _prob * _candidate_token_prob)

	# Here key-value cache is None temporary
	# I have to seek a way to update k, v cache
	with open("candidates.txt", 'w', encoding="utf8") as f:
		pass
	for i in range(max_length):		
		candidates = list()
		_easy_beam_search(_depth = depth, 
						  _inputs = inputs,
						  _past_key_values = past_key_values,
						  _prob = 1.,
						  )
		with open("candidates.txt", 'a', encoding="utf8") as f:
			f.write(f"Round: {i}, {len(candidates)}" + '\n')
			for candidate in candidates:
				f.write("{}\t{}\n".format(candidate[0], candidate[1]))
			f.write('-' * 4 + '\n')
		
		optimal_candidate_index = max(range(len(candidates)), key=lambda _i: candidates[_i][1])
		optimal_inputs = candidates[optimal_candidate_index][0]	# Long(1, n_tokens + i + depth)
		next_token_id = optimal_inputs[n_tokens + i].unsqueeze(-1)	# Long(1, )
		inputs = torch.cat([inputs, next_token_id.unsqueeze(-1)], dim=-1)	# Long(1, n_tokens + i) => Long(1, n_tokens + i + 1) 
	generated_text = tokenizer.decode(inputs[0], skip_special_tokens=True)	# Long(n_tokens + 1, ) => Str 
	return generated_text