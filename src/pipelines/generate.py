# -*- coding: utf8 -*-
# @author: caoyang
# @email: caoyang@stu.sufe.edu.cn

import torch
import pandas
import logging
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

from src.tools.transformers import greedy_decode, k_step_greedy_decode, beam_search_decode, generate_token_prob
from src.tools.hook import register_forward_hook_decorator, register_backward_hook_decorator

# @param tokenizer: Huggingface tokenizer Object
# @param text: [Str] Final generated text
# @param token_prob: List[Tuple(Int, Str, Float)], `len(generated_id_prob)` is `max_length`, indicating the generated probability of each token
# @param logits: Tuple[FloatTensor(1, n_vocab)], `len(generated_logits)` is `max_length`, indicating the logits when each token is generated
# @param k: [Int] top-k decode candidates to be display
# @param eos_id: [Int] tokenId of <eos> token, e.g. 151643(<|endoftext|>) for Qwen model 
def display_pipeline(tokenizer,
					 text,
					 token_probs,
					 logits,
					 k = 3,
					 eos_id = 151643,
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
		eos_prob = tensor_to_prob[eos_id].item()
		df_display["max_id"].append(max_id)
		df_display["cand_tokens"].append(cand_tokens)
		df_display["cand_probs"].append(probs)
		df_display["eos_prob"].append(eos_prob)
	df_display = pandas.DataFrame(df_display, columns=["max_id", "cand_tokens", "cand_probs", "eos_prob"])
	return pandas.concat([df_token_probs, df_display], axis=1)


# @param model_name_or_path: [Str]
# @param prompt: [Str]
# @param max_length: [Int]
# @param eos_id: [Int] default 151643 referes to <|endoftext|> of Qwen-xxx
# @param k: [Int] the number of top-k tokens to display
# @param device: [Str/torch.device] e.g. "cuda", "cpu", torch.device("cpu")
# @param generate_kwargs: [Dict] keyword arguments for `model.generate`
def generate_pipeline(model_name_or_path,
					  prompt,
					  max_length,
					  device = None,
					  generate_kwargs = None,
					  ):
	logging.info("Load model and tokenizer ...")
	tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
	model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)
	if device is None:
		device = "cuda" if torch.cuda.is_available() else "cpu"
	logging.info(f"Device: {device}")
	logging.info("Model Generate ...")
	if generate_kwargs is None:
		# Greedy decode configurations
		generate_kwargs = {"do_sample": False, "top_k": 0, "top_p": 1., "num_beams": 1, "temperature": 1}
	text, token_prob, logits = generate_token_prob(model, tokenizer, prompt, max_length, generate_kwargs, device)
	logging.info(f"Generated text: {text}")
	return display_pipeline(tokenizer, text, token_prob, logits)


def decode_pipeline(model_name_or_path,
					prompt,
					max_length,
					device = None,
					use_kv_cache = True,
					):
	logging.info("Load model and tokenizer ...")
	tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
	model = AutoModelForCausalLM.from_pretrained(model_name_or_path, trust_remote_code=True)
	if device is None:
		device = "cuda" if torch.cuda.is_available() else "cpu"
	logging.info(f"Device: {device} - KV Cache: {use_kv_cache}")
	logging.info("Greedy decode ...")
	returned_dict = greedy_decode(
		model = model,
		tokenizer = tokenizer,
		prompt = prompt,
		max_length = max_length,
		device = device,
		use_kv_cache = use_kv_cache,
		forward_hook_module_names = None,
		backward_hook_module_names = None,
	)
	text, token_probs, logits = returned_dict["text"], returned_dict["token_probs"], returned_dict["logits"]
	logging.info(f"Generated text: {text}")
	
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

	# logging.info("K step greedy decode ...")
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
	return display_pipeline(tokenizer, text, token_probs, logits)
