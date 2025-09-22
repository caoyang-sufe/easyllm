# -*- coding: utf8 -*-
# @author: caoyang
# @email: caoyang@stu.sufe.edu.cn
# Overwrite according to /transformers/models/llama/modeling_llama.py
# Version transformers 4.56.1

import torch
import logging
from torch import nn
from transformers import LlamaModel, LlamaForCausalLM
from transformers.cache_utils import Cache, DynamicCache
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple

class ParallelLlamaModel(LlamaModel):
	def __init__(self, config, n_cuda = 2, **kwargs):
		super().__init__(config, **kwargs)
		self.n_cuda = n_cuda

	def module_to_device(self):
		self.embed_tokens = self.embed_tokens.to("cuda:0")
		self.norm = self.norm.to(f"cuda:{self.n_cuda - 1}")
		n_layers = len(self.layers)
		self.layer_to_device = dict()
		for layer_id in range(n_layers):
			device_id = layer_id * self.n_cuda // n_layers
			self.layers[layer_id] = self.layers[layer_id].to(f"cuda:{device_id}")
			self.layer_to_device[layer_id] = device_id
			logging.info(f"Layer {layer_id} moved to cuda:{device_id}")

	def forward(self,
				input_ids = None,
				attention_mask = None,
				position_ids= None,
				past_key_values = None,
				inputs_embeds = None,
				cache_position = None,
				use_cache = None,
				**kwargs,
				):
		if (input_ids is None) ^ (inputs_embeds is not None):
			raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
		if inputs_embeds is None:
			inputs_embeds = self.embed_tokens(input_ids)
		if use_cache and past_key_values is None:
			past_key_values = DynamicCache(config=self.config)
		if cache_position is None:
			past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
			cache_position = torch.arange(
				past_seen_tokens, 
				past_seen_tokens + inputs_embeds.shape[1], 
				device = inputs_embeds.device,
			)
		if position_ids is None:
			position_ids = cache_position.unsqueeze(0)
		causal_mask = create_causal_mask(
			config=self.config,
			input_embeds=inputs_embeds,
			attention_mask=attention_mask,
			cache_position=cache_position,
			past_key_values=past_key_values,
			position_ids=position_ids,
		)
		hidden_states = inputs_embeds
		position_embeddings = self.rotary_emb(hidden_states, position_ids)
		# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
		for layer_id, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
			current_device_id = self.layer_to_device[layer_id]
			hidden_states = decoder_layer(
				hidden_states,
				attention_mask=causal_mask_mapping[decoder_layer.attention_type],
				position_ids=position_ids,
				past_key_values=past_key_values,
				cache_position=cache_position,
				position_embeddings=position_embeddings,
				**kwargs,
			)
			if layer_id < self.config.num_hidden_layers - 1:
				next_device_id = self.layer_to_device[layer_id + 1]
				if not current_device_id == next_device_id:
					next_device_name = f"cuda:{next_device_id}"
					hidden_states = hidden_states.to(next_device_name)
					if attention_mask is not None:
						attention_mask = causal_mask_mapping[decoder_layer.attention_type].to(next_device_name)
					if position_ids is not None:
						position_ids = position_ids.to(next_device_name)
					if cache_position is not None:
						cache_position = cache_position.to(next_device_name)
					# Deal with KV-Cache
					if past_key_values is not None:
						for i in range(past_key_values):
							if past_key_values[i][0] is not None:
								past_key_values[i][0] = past_key_values[i][0].to(next_device_name)
							if past_key_values[i][1] is not None:
								past_key_values[i][1] = past_key_values[i][1].to(next_device_name)
		hidden_states = self.norm(hidden_states).to("cuda:0")
		# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
		hidden_states = self.norm(hidden_states)
		return BaseModelOutputWithPast(
			last_hidden_state = hidden_states,
			past_key_values = past_key_values,
		)

class ParallelLlamaForCausalLM(LlamaForCausalLM):
	def __init__(self, config, n_cuda = 2):
		super(LlamaForCausalLM, self).__init__(config)
		self.model = ParallelLlamaModel(config, n_cuda = n_cuda)
		self.vocab_size = config.vocab_size
		self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
		# Initialize weights and apply final processing
		self.post_init()
		
	def module_to_device(self):
		self.model.module_to_device()
		# LM_HEAD need not be allocated to CUDA:1 because self.lm_head is equal to 
		# That is to say: `id(self.lm_head) == id(self.model.embed_tokens)`
		# self.lm_head = self.lm_head.to(f"cuda:{self.n_cuda - 1}")
