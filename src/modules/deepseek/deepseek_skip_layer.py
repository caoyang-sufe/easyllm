# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@stu.sufe.edu.cn
# Overwrite according to /transformers/models/qwen2/modeling_qwen2.py
# Version transformers 4.56.1

import torch
from torch import nn
from transformers.cache_utils import Cache, DynamicCache

from src.modules.deepseek import DeepseekModel, DeepseekForCausalLM

class SkipLayerDeepseekModel(DeepseekModel):
	def __init__(self, config, skip_layer_ids):
		super(DeepseekModel, self).__init__(config)
		self.skip_layer_ids = skip_layer_ids[:]
		
	def forward(self, *args, **kwargs):
		if self.skip_layer_ids:
			# 1. Delete `self.layers` and modify `layer.self_attn.layer_idx`
			filtered_layers = list()
			backup_layer_ids = list()
			backup_layers = self.layers
			for layer_id, layer in enumerate(self.layers):
				if layer_id not in self.skip_layer_ids:
					backup_layer_ids.append(layer_id)
					layer.self_attn.layer_idx = len(filtered_layers)
					filtered_layers.append(layer)
			self.layers = torch.nn.ModuleList(filtered_layers)
			# 2. Minus `self.config.num_hidden_layers`
			self.config.num_hidden_layers -= len(self.skip_layer_ids)
		result = super(SkipLayerQwen2Model, self).forward(*args, **kwargs)
		# Recover for follow-up callback
		if self.skip_layer_ids:
			# 1. Recover `self.layers`
			assert len(backup_layer_ids) == len(filtered_layers)
			for back_up_layer_id, layer in zip(backup_layer_ids, filtered_layers):
				layer.self_attn.layer_idx = back_up_layer_id
			self.layers = backup_layers	
			# 2. Recover `self.config.num_hidden_layers`
			self.config.num_hidden_layers += len(self.skip_layer_ids)
		return result


class SkipLayerDeepSeekForCausalLM(DeepseekForCausalLM):
	def __init__(self, config, skip_layer_ids):
		super(DeepseekModel, self).__init__(config)
		self.model = SkipLayerDeepseekModel(config, skip_layer_ids)
		self.vocab_size = config.vocab_size
		self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
		# Initialize weights and apply final processing
		self.post_init()
