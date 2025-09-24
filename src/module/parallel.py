# -*- coding: utf8 -*-
# @author: caoyang
# @email: caoyang@stu.sufe.edu.cn

import torch
from torch.nn import Module
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

class ParallelQwen2ForCausalLM(Module):
	
	def __init__(self, model_name_or_path, n_cuda = 2, **kwargs):
		super(ParallelQwen2ForCausalLM, self).__init__()
		self.model = AutoModelForCausalLM.from_pretrained(
			model_name_or_path,
			device_map = None
		)
		self.model.model.embed_tokens = self.model.model.embed_tokens.to("cuda:0")
		self.model.model.norm = self.model.model.norm.to(f"cuda:{n_cuda - 1}")
		self.model.model.lm_head = self.model.lm_head.to(f"cuda:{n_cuda - 1}")
		n_layers = len(self.layers)
		for i in range(n_layers):
			for j in range(n_cuda):
				if i <= (j + 1) * n_layers // n_cuda:
					self.model.model.layers[i] = self.model.model.layers[i].to(f"cuda: {j}")
					break

	def forward(self, *args, **kwargs):
		return self.model(*args, **kwargs)
