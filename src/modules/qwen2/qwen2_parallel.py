# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@stu.sufe.edu.cn
# Overwrite according to /transformers/models/qwen2/modeling_qwen2.py
# Version transformers 4.56.1

import torch
import logging
from torch import nn
from transformers import Qwen2Model, Qwen2ForCausalLM
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import TransformersKwargs, auto_docstring, can_return_tuple

class ParallelQwen2Model(Qwen2Model):
	def __init__(self, config, n_cuda = 2, **kwargs):
		super().__init__(config, **kwargs)
		self.n_cuda = n_cuda
		self.device_list = ["cpu", "cuda"] if self.n_cuda == 1 else [f"cuda:{i}" for i in range(n_cuda)]
		self.n_device = len(self.device_list)
		self.is_parallelizable = True
		self.model_parallel = True		
		self.module_to_device_flag = False

	def module_to_device(self):
		self.embed_tokens = self.embed_tokens.to(self.device_list[0])
		self.norm = self.norm.to(self.device_list[-1])
		n_layers = len(self.layers)
		self.layer_to_device = dict()
		for layer_id in range(n_layers):
			device_id = layer_id * self.n_device // n_layers
			self.layers[layer_id] = self.layers[layer_id].to(self.device_list[device_id])
			self.layer_to_device[layer_id] = device_id
			logging.info(f"Layer {layer_id} moved to {self.device_list[device_id]}")

	def forward(self,
				input_ids = None,
				attention_mask = None,
				position_ids= None,
				past_key_values = None,
				inputs_embeds = None,
				use_cache = None,
				cache_position = None,
				**kwargs,
				):
		# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
		if not self.module_to_device_flag:
			logging.info("First forward: move to device ...")
			self.module_to_device()
			self.module_to_device_flag = True
		input_ids = input_ids.to(self.device_list[0])
		if position_ids is not None:
			position_ids = position_ids.to(self.device_list[0])
		if cache_position is not None:
			cache_position = cache_position.to(self.device_list[0])
		# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
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
		# It may already have been prepared by e.g. `generate`
		if not isinstance(causal_mask_mapping := attention_mask, dict):
			# Prepare mask arguments
			mask_kwargs = {
				"config": self.config,
				"input_embeds": inputs_embeds,
				"attention_mask": attention_mask,
				"cache_position": cache_position,
				"past_key_values": past_key_values,
				"position_ids": position_ids,
			}
			# Create the masks
			causal_mask_mapping = {
				"full_attention": create_causal_mask(**mask_kwargs),
			}
			# The sliding window alternating layers are not always activated depending on the config
			if self.has_sliding_layers:
				causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)
		hidden_states = inputs_embeds
		# create position embeddings to be shared across the decoder layers
		position_embeddings = self.rotary_emb(hidden_states, position_ids)
		# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
		for layer_id, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
			current_device_id = self.layer_to_device[layer_id]
			current_device = self.device_list[current_device_id]
			hidden_states = decoder_layer(
				hidden_states,
				attention_mask = None if causal_mask_mapping[decoder_layer.attention_type] is None \
					else causal_mask_mapping[decoder_layer.attention_type].to(current_device),
				position_ids = position_ids,
				past_key_values = past_key_values,
				use_cache = use_cache,
				cache_position = cache_position,
				position_embeddings = position_embeddings,
				**kwargs,
			)
			if layer_id < self.config.num_hidden_layers - 1:
				next_device_id = self.layer_to_device[layer_id + 1]
				if not current_device_id == next_device_id:
					next_device_name = self.device_list[next_device_id]
					hidden_states = hidden_states.to(next_device_name)
					if position_ids is not None:
						position_ids = position_ids.to(next_device_name)
					if cache_position is not None:
						cache_position = cache_position.to(next_device_name)
					# Need not deal with KV-Cache
					# # Deal with KV-Cache
					# if past_key_values is not None:
						# new_past_key_values = DynamicCache(config=self.config)
						# for i in range(len(past_key_values)):
							# if not (past_key_values[i][0] is None and past_key_values[i][1] is None):
								# assert past_key_values[i][0] is not None and past_key_values[i][1] is not None
								# new_past_key_values.update(
									# key_states = past_key_values[i][0].to(next_device_name),
									# value_states = past_key_values[i][1].to(next_device_name),
									# layer_idx = i,
								# )
						# past_key_values = new_past_key_values
					# Deal with PostionEmbedding
					if position_embeddings is not None:
						# position_embeddings = (cos, sin)
						position_embeddings = (position_embeddings[0].to(next_device_name), position_embeddings[1].to(next_device_name))
		# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
		hidden_states = self.norm(hidden_states)
		return BaseModelOutputWithPast(
			last_hidden_state = hidden_states,
			past_key_values = past_key_values if use_cache else None,
		)

class ParallelQwen2ForCausalLM(Qwen2ForCausalLM):
	_tied_weights_keys = ["lm_head.weight"]
	_tp_plan = {"lm_head": "colwise_rep"}
	_pp_plan = {"lm_head": (["hidden_states"], ["logits"])}
	def __init__(self, config, n_cuda = 2):
		super(Qwen2ForCausalLM, self).__init__(config)
		self.model = ParallelQwen2Model(config, n_cuda = n_cuda)
		self.vocab_size = config.vocab_size
		self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
		# Initialize weights and apply final processing
		self.post_init()
		self.is_parallelizable = True
		self.model_parallel = True
		
	def module_to_device(self):
		self.model.module_to_device()
		self.lm_head = self.lm_head.to(self.model.device_list[0])
		# LM_HEAD need not be allocated to CUDA:1 because self.lm_head is equal to 
		# That is to say: `id(self.lm_head) == id(self.model.embed_tokens)`

	@can_return_tuple
	@auto_docstring
	def forward(
		self,
		input_ids = None,
		attention_mask = None,
		position_ids = None,
		past_key_values = None,
		inputs_embeds = None,
		labels = None,
		use_cache = None,
		cache_position = None,
		logits_to_keep = 0,
		**kwargs,
	):
		outputs = self.model(
			input_ids=input_ids,
			attention_mask=attention_mask,
			position_ids=position_ids,
			past_key_values=past_key_values,
			inputs_embeds=inputs_embeds,
			use_cache=use_cache,
			cache_position=cache_position,
			**kwargs,
		)
		
		hidden_states = outputs.last_hidden_state.to(self.model.device_list[0])	# <<<<<<<< Because `lm_head` must be on CUDA:0 according to `embed_tokens` >>>>>>>>
		# Only compute necessary logits, and do not upcast them to float if we are not computing the losse
		slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
		logits = self.lm_head(hidden_states[:, slice_indices, :])
		loss = None
		if labels is not None:
			# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
			loss = self.loss_function(logits=logits, labels=labels.to(self.model.device_list[-1]), vocab_size=self.config.vocab_size, **kwargs)
			# <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
		return CausalLMOutputWithPast(
			loss=loss,
			logits=logits,
			past_key_values=outputs.past_key_values,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
		)
