# -*- coding: utf8 -*-
# @author: caoyang
# @email: caoyang@stu.sufe.edu.cn

import torch
import logging
from copy import deepcopy
from functools import wraps
from torch.nn import functional as F

from transformers import AutoTokenizer, AutoModelForCausalLM
from src.tools.hook import register_forward_hook_decorator, register_backward_hook_decorator

def get_generation_eos_token_ids(model):
	eos_token_id = model.generation_config.eos_token_id
	if isinstance(eos_token_id, int):
		eos_token_ids = [eos_token_id]
	elif isinstance(eos_token_id, list):
		eos_token_ids = eos_token_id[:]
	else:
		logging.warning(f"Unknown type of EOS: {eos_token_id}")
		eos_token_ids = [151643, 151645]	# Default EOS token for Qwen model
	return eos_token_ids

def get_generation_bos_token_ids(model):
	bos_token_id = model.generation_config.bos_token_id
	if isinstance(bos_token_id, int):
		bos_token_ids = [bos_token_id]
	elif isinstance(bos_token_id, list):
		bos_token_ids = bos_token_id[:]
	else:
		logging.warning(f"Unknown type of BOS: {bos_token_id}")
		bos_token_ids = [151646]	# Default EOS token for Qwen model
	return bos_token_ids	

# Standard greedy decode supporting hook registration
# @param model: Huggingface model object
# @param tokenizer: Huggingface tokenizer Object
# @param prompt: [Str]
# @param max_length: [Int]the number of tokens to be generated (exclude `prompt`)
# @param device: [Str] e.g. "cuda" or "cpu"
# @param use_kv_cache: [Boolean] whether to use KV-cache to accelerate, if True then large memory will be consumed
# @param forward_hook_module_names: [List[Str]] Default None, otherwise register forward hook for `forward_hook_module_names`, e.g. ["model.layers[0].self_attn.q_proj", "model.layers[0].self_attn.k_proj"]
# @param backward_hook_module_names: [List[Str]] Default None, otherwise register backward hook for `backward_hook_module_names`, e.g. ["model.layers[0].self_attn.q_proj", "model.layers[0].self_attn.k_proj"]
# @return returned_dict: [Dict]
# - "text": [Str]
# - "token_probs": List[Tuple(Int, Str, Float)], `len(generated_id_prob)` is `max_length`, indicating the generated probability of each token
# - "logits": Tuple[FloatTensor(1, n_vocab)], `len(generated_logits)` is `max_length`, indicating the logits when each token is generated
# - "forward_hook_data": List of hook data (length is `max_length`) in format of [Dict[<module_name>: Dict]], Read `hook_data` in `src.tools.hook.register_forward_hook_decorator` for details
# - "backward_hook_data": List of hook data (length is `max_length`) in format of [Dict[<module_name>: Dict]], Read `hook_data` in `src.tools.hook.register_backward_hook_decorator` for details
def greedy_decode(model,
				  tokenizer,
				  prompt, 
				  max_length,
				  device = "cuda",
				  use_kv_cache = True,
				  forward_hook_module_names = None,
				  backward_hook_module_names = None,
				  ):
	if forward_hook_module_names is None and backward_hook_module_names is None:
		hook_flag = 0
		def easy_forward(inputs, *, model, **kwargs):
			return model(inputs, **kwargs)
	elif not (forward_hook_module_names is None or backward_hook_module_names is None):
		hook_flag = -1
		raise NotImplementedError("Simultaneous use of forward and backward hook!")
	else:
		if forward_hook_module_names is not None:
			hook_flag = 1
			@register_forward_hook_decorator(module_names = forward_hook_module_names)
			def easy_forward(inputs, *, model, **kwargs):
				return model(inputs, **kwargs)
		else:
			hook_flag = 2
			@register_backward_hook_decorator(module_names = forward_hook_module_names)
			def easy_forward(inputs, *, model, **kwargs):
				return model(inputs, **kwargs)			

	eos_token_ids = get_generation_eos_token_ids(model)
	logging.info(f"EOS token: {eos_token_ids}")
	inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)	# Str => Long(1, n_tokens)
	past_key_values = None
	generated_token_probs = list()
	generated_logits = list()
	forward_hook_data = None
	backward_hook_data = None
	if hook_flag == 1:
		forward_hook_data = list()
	if hook_flag == 2:
		backward_hook_data = list()	
	for i in range(max_length):
		logging.info(f"Round {i}: {len(past_key_values) if past_key_values is not None else None}")
		with torch.no_grad():
			if use_kv_cache:
				if past_key_values is None:
					outputs = easy_forward(inputs, model=model, past_key_values=None, use_cache=True)
				else:
					outputs = easy_forward(inputs[:, -1].unsqueeze(0), model=model, past_key_values=past_key_values, use_cache=True)
				past_key_values = outputs.past_key_values	# Dictlike[key_cache: Float(1, 2, X, hidden_size), value_cache: Float(1, 2, X, hidden_size)], where X = (i + 1) * (n_tokens + i / 2)
			else:
				outputs = easy_forward(inputs, model=model, past_key_values=None, use_cache=False)
			logits = outputs.logits	# Float(1, n_tokens + i + 1, n_vocab), where `n_vocab` is 151936 in Qwen-series
			next_token_probs = F.softmax(logits[:, -1, :], dim=-1)	# Float(1, n_tokens + i + 1, n_vocab) => Float(1, n_vocab)
			next_token_id = torch.argmax(next_token_probs, dim=-1)	# Float(1, n_vocab) => Long(1, )
			next_token_prob = next_token_probs[0, next_token_id].item()	# Float(1, n_vocab) => Float()
			next_token = tokenizer.decode(next_token_id[0].item(), skip_special_tokens=False)	# Long(1, ) => Str
			inputs = torch.cat([inputs, next_token_id.unsqueeze(-1)], dim=-1)	# Long(1, n_tokens + i) => Long(1, n_tokens + i + 1)
			generated_token_probs.append((next_token_id.item(), next_token, next_token_prob))	# List[] <- (Int, Str, Float)
			generated_logits.append(logits[:, -1, :])	# List[] <- Float(1, n_vocab)
			# Process hook data
			if hook_flag > 0:
				hook_data = outputs.hook_outputs
				if hook_flag == 1:
					forward_hook_data.append(hook_data)
				else:
					backward_hook_data.append(hook_data)
			if next_token_id in eos_token_ids:
				# Early stop at EOS
				break
	generated_text = tokenizer.decode(
		token_ids = inputs[0], 
		skip_special_tokens=True, 
		clean_up_tokenization_spaces=True,
	)	# Long(1, n_tokens + max_length) => Str
	return {
		"text": generated_text,
		"token_probs": generated_token_probs,
		"logits": tuple(generated_logits),
		"forward_hook_data": forward_hook_data,
		"backward_hook_data": backward_hook_data,
	}

# K-step greedy decode
# @param model: Huggingface model object
# @param tokenizer: Huggingface tokenizer Object
# @param prompt: [Str]
# @param max_length: [Int] the number of tokens to be generated (exclude `prompt`)
# @param n_branches: [Int] the number of branches searched each step 
# @param depth: [Int] the number of step searched
# @param device: [Str] e.g. "cuda" or "cpu"
# @param use_kv_cache: [Boolean] whether to use KV-cache to accelerate, currently we do not support KV-cache here
# @return generated_text: [Str]
def k_step_greedy_decode(model,
						 tokenizer,
						 prompt,
						 max_length,
						 n_branches,
						 depth,
						 device = "cuda",
						 use_kv_cache = False,
						 ):
	# Here key-value cache is None temporary
	# I have to seek a way to update k, v cache
	if use_kv_cache:
		logging.warning("`use_kv_cache = True` is not implemented for `k_step_greedy_decode`")
		use_kv_cache = False
	inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)	# Str => Long(1, n_tokens)
	n_tokens = inputs.size(1)
	past_key_values = None
	def _easy_dp(_depth, _inputs, _past_key_values, _prob):
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
				_easy_dp(_depth - 1, _candidate_inputs, _past_key_values, _prob * _candidate_token_prob)
	for i in range(max_length):		
		candidates = list()	# List[Tuple(Long(1, n_tokens + i + depth), Float)]
		_easy_dp(_depth = depth, 
				 _inputs = inputs,
				 _past_key_values = past_key_values,
				 _prob = 1.,
				 )
		logging.info(f"Round {i}: {len(candidates)} candidates")
		for candidate in candidates:
			logging.debug(f"- {candidate[0].size()} - {candidate[1]}")	# Shape & probability
		# Explanation:
		# - Take the candidate inputs with largest probability in `candidates` => `optimal_candidate_index`
		# - Generated the optimal result in `depth` step forward => `optimal_inputs`
		# - But take only the next token in `optimal_inputs` => `next_token_id`
		optimal_candidate_index = max(range(len(candidates)), key=lambda _i: candidates[_i][1])
		optimal_inputs = candidates[optimal_candidate_index][0]	# Long(n_tokens + i + depth, )
		next_token_id = optimal_inputs[n_tokens + i].unsqueeze(-1)	# Long(1, )
		inputs = torch.cat([inputs, next_token_id.unsqueeze(-1)], dim=-1)	# Long(1, n_tokens + i) => Long(1, n_tokens + i + 1) 
	generated_text = tokenizer.decode(inputs[0], skip_special_tokens=True)	# Long(n_tokens + 1, ) => Str 
	return generated_text

# Beam search decode with KV-Cache
# @param model: Huggingface model object
# @param tokenizer: Huggingface tokenizer Object
# @param prompt: [Str]
# @param max_length: [Int] the number of tokens to be generated (exclude `prompt`)
# @param length_penalty: [Int] larger penalty value refers to preference to long sequence
# @param device: [Str] e.g. "cuda" or "cpu"
# @param use_kv_cache: [Boolean], whether to use KV-cache to accelerate, if True then large memory will be consumed
# @return generated_texts: List[Str] of length num_beams
def beam_search_decode(model, 
					   tokenizer,
					   prompt,
					   max_length,
					   num_beams,
					   length_penalty = 1.,
					   device = "cuda",
					   use_kv_cache = True,
					   ):
	eos_token_ids = get_generation_eos_token_ids(model)
	logging.info(f"EOS token: {eos_token_ids}")
	inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)	# Str => Long(1, n_tokens)
	sequences = [inputs.tolist()[0]]  # List[[List[Int]]]
	scores = [0.]   # List[Float]
	kv_caches = [None]  # <kv-cache object>: List[Dictlike[key_cache: FloatTensor, value_cache: FloatTensor]]
	completed_sequences = []
	completed_scores = []
	for i in range(max_length):
		candidates = [] # List[Tuple[List[Int], List[Float], <kv-cache object>]]
		new_kv_caches = []
		for sequence, score, kv_cache in zip(sequences, scores, kv_caches):
			# Skip sequences which encounted EOS token
			if len(sequence) > 0 and sequence[-1] in eos_token_ids:
				completed_sequences.append(sequence)
				completed_scores.append(score)
				continue
			with torch.no_grad():
				if use_kv_cache:
					if kv_cache is None:
						# First round: kv_cache is None
						inputs = torch.tensor([sequence], dtype=torch.long).to(device)
						outputs = model(inputs, use_cache=True)
					else:
						# Need only the last token because the former is cached
						inputs = torch.tensor([[sequence[-1]]], dtype=torch.long).to(device)  
						outputs = model(inputs, past_key_values=kv_cache, use_cache=True)
					new_kv_cache = outputs.past_key_values
				else:
					inputs = torch.tensor([sequence], dtype=torch.long).to(device)
					outputs = model(inputs, use_cache=False)
					new_kv_cache = None
				next_token_logits = outputs.logits[:, -1, :]	# Float(1, n_tokens + i + 1, n_vocab) => Float(1, n_vocab)
				next_token_probs = F.log_softmax(next_token_logits, dim=-1)	# Float(1, n_vocab) => Float(1, n_vocab)
			top_k_probs, top_k_tokens = torch.topk(next_token_probs, k=num_beams, dim=-1)	# Float(1, n_vocab) => Float(1, n_vocab), Long(1, n_vocab)
			top_k_probs, top_k_tokens = top_k_probs.squeeze(0), top_k_tokens.squeeze(0)	# Float(1, n_vocab), Long(1, n_vocab) => Float(n_vocab, ), Long(n_vocab, )
			for j in range(num_beams):
				new_sequence = sequence.copy()	# List of length(n_tokens)
				new_sequence.append(top_k_tokens[j].item())	# List[Int]: Add candidate token
				new_score = score + top_k_probs[j].item()	# List[Float]: Accumulate logprobs
				candidates.append((new_sequence, new_score, new_kv_cache))	# List[Tuple[List[Int], List[Float], <kv-cache object>]]
		if not candidates:
			# No candidates then break
			break
		candidates.sort(key=lambda x: x[1] / (len(x[0]) ** length_penalty), reverse=True)
		top_k_sequences, top_k_scores, top_k_kv_caches = map(list, zip(*candidates[: num_beams]))	# Prune to top-k candidates
	completed_sequences.extend(top_k_sequences)
	completed_scores.extend(top_k_scores)
	sorted_pairs = sorted(zip(completed_sequences, completed_scores), key=lambda x: x[1] / (len(x[0]) ** length_penalty), reverse=True)
	sorted_sequences, sorted_scores = zip(*sorted_pairs)
	generated_texts = tokenizer.batch_decode(sorted_sequences, skip_special_tokens=True, clean_up_tokenization_spaces=False)
	return generated_texts

# Get the output probability of generated token by `model.generate`
# @param model: Huggingface model object
# @param tokenizer: Huggingface tokenizer Object
# @param prompt: [Str]
# @param max_length: [Int] the number of tokens to be generated (include `prompt`)
# @param generate_kwargs: [Dict] Keyword arguments for `model.generate`, e.g. strict greedy decode <=> {"do_sample": False, "top_k": 0, "top_p": 1., "num_beams": 1, "temperature": 1}
# @param device: [Str] e.g. "cuda" or "cpu"
# @return generated_text: [Str]
# @return generated_token_prob: List[Tuple(Int, Str, Float)] `len(generated_id_prob)` is `max_length - <prompt_length>`, indicating the generated probability of each token
# @return generated_logits: Tuple[FloatTensor(1, n_vocab)] `len(generated_logits)` is `max_length - <prompt_length>`, indicating the logits when each token is generated
def generate_token_prob(model, 
						tokenizer, 
						prompt, 
						max_length,
						generate_kwargs = {"do_sample": False, "top_k": 0, "top_p": 1., "num_beams": 1, "temperature": 1},
						device = "cuda",
						):
	inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
	with torch.no_grad():
		outputs = model.generate(inputs, max_length=max_length, output_scores=True, return_dict_in_generate=True, **generate_kwargs)
		generated_token_ids = outputs.sequences	# Long(1, max_length)
		generated_logits = outputs.scores	# Tuple(Float(1, n_vocab)) with length (max_length - n_tokens)
		generated_probs = tuple(map(lambda _logits: F.softmax(_logits, dim=-1), generated_logits))
		generated_text = tokenizer.batch_decode(
			generated_token_ids,
			skip_special_tokens=True,
			clean_up_tokenization_spaces=False,
		)[0]
		generated_token_probs = list()
		diff = generated_token_ids.size(1) - len(generated_probs)
	for i in range(len(generated_probs)):
		token_id = generated_token_ids[0, i + diff].item()	# Int
		token = tokenizer.decode(token_id, skip_special_tokens=False)	# Str
		token_prob = generated_probs[i][0, token_id].item()	# Float
		generated_token_probs.append((token_id, token, token_prob))
	return generated_text, generated_token_probs, generated_logits

# Calculate cosine similarity by filtering outlier
# @param x: [torch.Tensor]
# @param y: [torch.Tensor]
# @param filter_outlier: [Float] range from [0, 1)
def robust_cosine_similarity(x, y, outlier_ratio = .1):
	x, y = x.flatten(), y.flatten()
	assert x.size(0) == y.size(0)
	abs_diff = torch.abs(x - y)
	k = int(len(abs_diff) * (1 - outlier_ratio))
	_, indices = torch.topk(abs_diff, k=k, largest=False)
	x_filtered, y_filtered = x[indices], y[indices]
	similarity = F.cosine_similarity(x_filtered, y_filtered, dim=0).item()
	return similarity
	

# Calculate correlation coefficient by filtering outlier
# @param x: [torch.Tensor]
# @param y: [torch.Tensor]
# @param outlier_ratio: [Float] range from [0, 1)
def robust_corrcoef(x, y, outlier_ratio = .1):
	x, y = x.flatten(), y.flatten()
	assert x.size(0) == y.size(0)
	abs_diff = torch.abs(x - y)
	k = int(len(abs_diff) * (1 - outlier_ratio))
	_, indices = torch.topk(abs_diff, k=k, largest=False)
	x_filtered, y_filtered = x[indices], y[indices]
	corrcoef = torch.corrcoef(torch.stack([x_filtered, y_filtered]))[0, 1].item()
	return corrcoef
