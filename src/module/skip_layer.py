# -*- coding: utf8 -*-
# @author: caoyang
# @email: caoyang@stu.sufe.edu.cn

import re
import torch
import logging
from matplotlib import pyplot as plt

from src.tools.plot import plot_tensor_heatmap
from src.tools.transformers import greedy_decode

def create_model_class(SuperModel):
	class SkipLayerModel(SuperModel):
		# @param config: AutoConfig object
		# @param skip_layer_ids: List[Int], Layer # to be skipped
		def __init__(self, config, skip_layer_ids = list(), **kwargs):
			super(SkipLayerModel, self).__init__(config, **kwargs)
			self.skip_layer_ids = skip_layer_ids
		def forward(self, *args, **kwargs):
			original_layers = self.layers
			if self.skip_layer_ids:
				filtered_layers = [
					layer for i, layer in enumerate(self.layers)
					if i not in self.skip_layer_ids
				]
				self.layers = torch.nn.ModuleList(filtered_layers)
			result = super(SkipLayerModel, self).forward(*args, **kwargs)
			self.layers = original_layers	# Recover for follow-up callback
			return result
	return SkipLayerModel

def create_model_class_for_causal_lm(SuperModel, SuperModelForCausalLM):
	Model = create_model_class(SuperModel)
	class SkipLayerModelForCausalLM(SuperModelForCausalLM):
		# @param config: AutoConfig object
		# @param Model: AutoModel class
		def __init__(self, config, skip_layer_ids = list()):
			super(SkipLayerModelForCausalLM, self).__init__(config)
			self.model = Model(config = config, skip_layer_ids = skip_layer_ids)	
	return SkipLayerModelForCausalLM
