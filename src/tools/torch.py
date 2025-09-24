# -*- coding: utf8 -*-
# @author: caoyang
# @email: caoyang@stu.sufe.edu.cn

import torch
from functools import wraps
from torch.nn import functional as F

# Forward hook decorator
# @params module_names: List[Str], e.g. ["model.layers[0].self_attn.q_proj", "model.layers[0].self_attn.k_proj"]
def register_forward_hook_decorator(module_names):
	# @param func: the call of `func` must include a keyword argument "model"
	def decorator(func):
		@wraps(func)
		def wrapper(*args, **kwargs):
			model = kwargs["model"]
			hook_data = dict()
			hook_handles = list()
			for module_name in module_names:
				def _make_hook(_module_name):
					hook_data[module_name] = dict()
					# @param _module: `f"Module: {_module.__class__.__name__}"`
					# @param _inputs: Tuple[torch.FloatTensor], `f"Input shapes: {[x.shape for x in _inputs]}"`
					# @param _outputs: torch.FloatTensor|Tuple[torch.FloatTensor], `f"Output shape: {_outputs.shape}"`
					def _hook(_module, _input, _output):
						hook_data[_module_name]["output"] = _output
						hook_data[_module_name]["input"] = _input
					return _hook
				hook_handles.append(eval(f"model.{module_name}").register_forward_hook(_make_hook(module_name)))
			try:
				func_return = func(*args, **kwargs)
				func_return.hook_outputs = hook_data	# Attach hook data to function returns
				return func_return
			finally:
				for hook_handle in hook_handles:
					hook_handle.remove()
		return wrapper
	return decorator

# Backward hook decorator (for gradient)
# @params module_names: List[Str], e.g. ["model.layers[0].self_attn.q_proj", "model.layers[0].self_attn.k_proj"]
def register_backward_hook_decorator(module_names):
	# @param func: the call of `func` must include a keyword argument "model"
	def decorator(func):
		@wraps(func)
		def wrapper(*args, **kwargs):
			model = kwargs["model"]
			hook_data = dict()
			hook_handles = list()
			for module_name in module_names:
				def _make_hook(_module_name):
					hook_data[module_name] = {"input": list(), "output": list()}
					# @param _module: `f"Module: {_module.__class__.__name__}"`
					# @param _inputs: Tuple[torch.FloatTensor], `f"Input shapes: {[x.shape for x in _inputs]}"`
					# @param _outputs: Tuple`f"Output shape: {_outputs.shape}"`
					def _hook(_module, _input, _output):
						hook_data[_module_name]["input"].append(_input.detach().clone())
						hook_data[_module_name]["output"].append(_output.detach().clone())
					return _hook
				hook_handles.append(eval(f"model.{module_name}").register_forward_hook(_make_hook(module_name)))
			try:
				func_return = func(*args, **kwargs)
				func_return.hook_outputs = hook_data	# Attach hook data to function returns
				return func_return
			finally:
				for hook_handle in hook_handles:
					hook_handle.remove()
		return wrapper
	return decorator


# Standard greedy decode with hook
# @param hooked_module_names: [List] e.g. ["model.layers[0].self_attn.q_proj", "model.layers[0].self_attn.k_proj"]
# @param model: Huggingface model object
# @param tokenizer: Huggingface tokenizer Object
# @param prompt: [Str]
# @param max_length: [Int]the number of tokens to be generated (exclude `prompt`)
# @param device: [Str] e.g. "cuda" or "cpu"
# @param use_kv_cache: [Boolean] whether to use KV-cache to accelerate, if True then large memory will be consumed
# @return generated_text: [Str]
# @return generated_token_prob: List[Tuple(Int, Str, Float)], `len(generated_id_prob)` is `max_length`, indicating the generated probability of each token
# @return generated_logits: Tuple[FloatTensor(1, n_vocab)], `len(generated_logits)` is `max_length`, indicating the logits when each token is generated
def greedy_decode_with_forward_hook(hooked_module_names,
									model,
									tokenizer,
									prompt, 
									max_length,
									device = "cuda",
									use_kv_cache = True,
									):
	inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)	# Str => Long(1, n_tokens)
	past_key_values = None
	generated_token_probs = list()
	generated_logits = list()

	@register_forward_hook_decorator(hooked_module_names)
	def _forward_with_multi_hooks(_model, _inputs, _past_key_values):
		_outputs = _model(_inputs, past_key_values=_past_key_values)
		return _outputs

	for i in range(max_length):
		logging.info(f"Round {i}: {past_key_values.key_cache[0].size() if past_key_values is not None else None}")
		outputs = model(_inputs, past_key_values=_past_key_values)
		logits = outputs.logits	# Float(1, n_tokens + i + 1, n_vocab), where `n_vocab` is 151936 in Qwen-series
		if use_kv_cache:
			if past_key_values is None:
				forward_with_multi_hooks(model, inpts)
			else:
				# outputs = model(inputs[:, -1].unsqueeze(0), past_key_values=past_key_values, use_cache=True)
				outputs = model(inputs, past_key_values=past_key_values)
			past_key_values = outputs.past_key_values	# Dictlike[key_cache: Float(1, 2, X, hidden_size), value_cache: Float(1, 2, X, hidden_size)], where X = (i + 1) * (n_tokens + i / 2)
		else:
			outputs = model(inputs, past_key_values=None, use_cache=False)
		next_token_probs = F.softmax(logits[:, -1, :], dim=-1)	# Float(1, n_tokens + i + 1, n_vocab) => Float(1, n_vocab)
		next_token_id = torch.argmax(next_token_probs, dim=-1)	# Float(1, n_vocab) => Long(1, )
		next_token_prob = next_token_probs[0, next_token_id].item()	# Float(1, n_vocab) => Float()
		next_token = tokenizer.decode(next_token_id[0].item(), skip_special_tokens=False)	# Long(1, ) => Str
		inputs = torch.cat([inputs, next_token_id.unsqueeze(-1)], dim=-1)	# Long(1, n_tokens + i) => Long(1, n_tokens + i + 1)
		generated_token_probs.append((next_token_id.item(), next_token, next_token_prob))	# List[] <- (Int, Str, Float)
		generated_logits.append(logits[:, -1, :])	# List[] <- Float(1, n_vocab)
	generated_text = tokenizer.decode(
		token_ids = inputs[0], 
		skip_special_tokens=True, 
		clean_up_tokenization_spaces=True,
	)	# Long(1, n_tokens + max_length) => Str
	return generated_text, generated_token_probs, tuple(generated_logits)
