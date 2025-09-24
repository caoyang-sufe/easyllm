# -*- coding: utf8 -*-
# @author: caoyang
# @email: caoyang@stu.sufe.edu.cn

import torch
from torch.nn import functional as F

# Calculate mean token accuracy
# @param target: [List[List[<token>]]] <token> is Int or Str (prefer Int)
# @param predict: [List[List[<token>]]] <token> is Int or Str (prefer Int)
# @return: [Float]
def calc_mean_token_accuracy(predict, target):
	correct = 0
	total = 0
	for predict_tokens, target_tokens in zip(predict, target):
		n_tokens = min(len(predict_tokens), len(target_tokens))
		if n_tokens == 0:
			continue
		correct += sum(1 for predict_token, target_token in zip(predict_tokens[: n_tokens], target_tokens[: n_tokens]) if predict_token == target_token)
		total += n_tokens
	return correct / total if total > 0 else 0

# Calculate perplexity on completion only
# @param prompt: [Str|List[<token_id>]] <token_id> is Int only
# @param completion: [Str|List[<token_id>]] <token_id> is Int only
# @param model: Huggingface model object
# @param tokenizer: Huggingface tokenizer object, which can be None when `prompt` and `completion` are List
# @param apply_chat_template: [Function] consist of two positional or keyword arguments: (prompt, completion), default `None` refers to simply concatenation
# @return: [Float]
def calc_perplexity(prompt, 
					completion, 
					model,
					tokenizer = None,
					apply_chat_template = None,
					):
	if apply_chat_template is None:
		prompt_token_ids = tokenizer.encode(prompt, return_tensors="pt") if isinstance(prompt, str) else prompt[:]
		completion_token_ids = tokenizer.encode(completion, return_tensors="pt") if isinstance(completion, str) else completion[:]
		prompt_token_length = prompt_token_ids.size(1)
		input_ids = torch.cat([prompt_token_ids, completion_token_ids], dim=-1)
		labels = input_ids.clone()
		labels[:, : prompt_token_length] = -100	
		with torch.no_grad():
			outputs = model(input_ids=input_ids, labels=labels)
			return torch.exp(outputs.loss).item()	
	else:
		raise NotImplementedError("Currently do not support `apply_chat_template`")
