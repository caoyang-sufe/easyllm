# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@stu.sufe.edu.cn

import torch
import pandas
import logging
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.tools.transformers import greedy_decode, k_step_greedy_decode, beam_search_decode, generate_token_prob, get_generation_eos_token_ids
from src.tools.hook import register_forward_hook_decorator, register_backward_hook_decorator
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

# Do only one time forward to check model outputs
# @param model: Huggingface AutoModel object
def one_time_forward_pipeline(
	model, 
	tokenizer,
	prompt,
	device = None,
	forward_hook_module_names = None,
	backward_hook_module_names = None,
):
	if backward_hook_module_names is not None:
		raise NotImplementedError("Currently not support `backward_hook_module_names`")
	@register_forward_hook_decorator(module_names = forward_hook_module_names)
	def easy_forward(inputs, *, model, **kwargs):
		return model(inputs, **kwargs)
	if device is None:
		device = "cuda" if torch.cuda.is_available() else "cpu"
	logging.info(f"Device: {device}")	
	inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)	# Str => Long(1, n_tokens)
	outputs = easy_forward(inputs, model=model, use_cache=False)
	hook_data = outputs.hook_outputs
	return hook_data


# Display generation details token by token
# @param tokenizer: Huggingface tokenizer Object
# @param text: [Str] Final generated text
# @param token_prob: List[Tuple(Int, Str, Float)], `len(generated_id_prob)` is `max_length`, indicating the generated probability of each token
# @param logits: Tuple[FloatTensor(1, n_vocab)], `len(generated_logits)` is `max_length`, indicating the logits when each token is generated
# @param k: [Int] top-k decode candidates to be display
# @param eos_token_id: [Int] tokenId of <eos> token, e.g. 151643(<|endoftext|>) for Qwen model
# @return df_display: [pandas.DataFrame] ["id", "token", "prob", "max_id", "cand_tokens", "cand_probs", "eos_prob"]
def display_pipeline(tokenizer,
					 text,
					 token_probs,
					 logits,
					 k = 3,
					 eos_token_id = 151643,
					 ):
	df_token_probs = pandas.DataFrame(token_probs, columns=["id", "token", "prob"])
	def _display_tensor(_tensor, _round):
		return list(map(lambda x: round(x, _round), _tensor.tolist()))
	df_display = {
		"max_id": [],
		"cand_tokens": [],
		"cand_probs": [],
		"eos_prob": [],
	}
	for tensor in logits:
		tensor_to_prob = F.softmax(tensor[0], dim=-1)
		top_k = torch.topk(tensor_to_prob, k = 3)
		top_k_values = top_k.values
		top_k_indices = top_k.indices
		max_id = top_k_indices[0].item()
		probs = _display_tensor(top_k_values, 4)
		cand_ids = _display_tensor(top_k_indices, 4)
		cand_tokens = [tokenizer.decode(token_id) for token_id in top_k_indices]
		eos_prob = tensor_to_prob[eos_token_id].item()
		df_display["max_id"].append(max_id)
		df_display["cand_tokens"].append(cand_tokens)
		df_display["cand_probs"].append(probs)
		df_display["eos_prob"].append(eos_prob)
	df_display = pandas.DataFrame(df_display, columns=["max_id", "cand_tokens", "cand_probs", "eos_prob"])
	return pandas.concat([df_token_probs, df_display], axis=1)

# Generate tokens by a given prompt, using `model.generate`
# @param model_name_or_path: [Str]
# @param prompt: [Str]
# @param max_length: [Int]
# @param device: [Str|torch.device] e.g. "cuda", "cpu", torch.device("cpu")
# @param generate_kwargs: [Dict] keyword arguments for `model.generate`
# @return df_display: the returned of `display_pipeline`
def generate_pipeline(model_name_or_path,
					  prompt,
					  max_length,
					  device = None,
					  generate_kwargs = None,
					  model_parallel_class = None,
					  n_cuda = 2,
					  ):
	logging.info("Load model and tokenizer ...")
	if device is None:
		device = "cuda" if torch.cuda.is_available() else "cpu"
	logging.info(f"Device: {device}")
	tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
	if model_parallel_class is None:
		model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True).to(device)
	else:
		model = eval(model_parallel_class).from_pretrained(model_name_or_path, n_cuda=n_cuda)
		model.module_to_device()
	eos_token_ids = get_generation_eos_token_ids(model)
	logging.info(f"  - EOS Tokens: {eos_token_ids}")
	logging.info("Model Generate ...")
	if generate_kwargs is None:
		# Greedy decode configurations
		generate_kwargs = {"do_sample": False, "top_k": 0, "top_p": 1., "num_beams": 1, "temperature": 1}
	text, token_prob, logits = generate_token_prob(model, tokenizer, prompt, max_length, generate_kwargs, device)
	logging.info(f"Generated text: {text}")
	return display_pipeline(tokenizer, text, token_prob, logits, eos_token_id=eos_token_ids[0])

# Generate tokens by a given prompt, using `src.tools.transformers`
# @param model_name_or_path: [Str]
# @param prompt: [Str]
# @param max_length: [Int]
# @param device: [Str|torch.device] e.g. "cuda", "cpu", torch.device("cpu")
# @param use_kv_cache: [Boolean]
# @param forward_hook_module_names: List[Str]
# @param backward_hook_module_names: List[Str]
# @return: Dict["df_display", "forward_hook_data", "backward_hook_data"]
def decode_pipeline(model_name_or_path,
					prompt,
					max_length,
					device = "cuda",
					use_kv_cache = True,
					forward_hook_module_names = None,
					backward_hook_module_names = None,
					):
	logging.info("Load model and tokenizer ...")
	if device is None:
		device = "cuda" if torch.cuda.is_available() else "cpu"
	logging.info(f"Device: {device} - KV Cache: {use_kv_cache}")
	tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
	model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True).to(device)
	eos_token_ids = get_generation_eos_token_ids(model)
	logging.info(f"  - EOS Tokens: {eos_token_ids}")
	logging.info("Greedy decode ...")
	returned_dict = greedy_decode(
		model = model,
		tokenizer = tokenizer,
		prompt = prompt,
		max_length = max_length,
		device = device,
		use_kv_cache = use_kv_cache,
		forward_hook_module_names = forward_hook_module_names,
		backward_hook_module_names = backward_hook_module_names,
	)
	text, token_probs, logits = returned_dict["text"], returned_dict["token_probs"], returned_dict["logits"]
	forward_hook_data, backward_hook_data = returned_dict["forward_hook_data"], returned_dict["backward_hook_data"]
	logging.info(f"Generated text: {text}")

	# # Beam decoding
	# logging.info("Beam decode ...")
	# beam_search_decode(
		# model = model,
		# tokenizer = tokenizer,
		# prompt = prompt,
		# max_length = max_length,
		# num_beams = 2,
		# length_penalty = 1.,
		# device = device,
		# use_kv_cache = use_kv_cache,
	# )

	# # K-step greedy decoding
	# logging.info("K step greedy decode ...") bn
	# k_step_greedy_decode(
		# model = model,
		# tokenizer = tokenizer,
		# prompt = prompt,
		# max_length = max_length,
		# n_branches = 2,
		# depth = 3,
		# device = device,
		# use_kv_cache = False,
	# )
	return {
		"df_display": display_pipeline(tokenizer, text, token_probs, logits, eos_token_id=eos_token_ids[0]),
		"forward_hook_data": forward_hook_data,
		"backward_hook_data": backward_hook_data,
	}
