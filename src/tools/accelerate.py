# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@stu.sufe.edu.cn

import torch
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from transformers import AutoModelForCausalLM

# Load model by device mapping
# @param model_path: [Str]
# @param Model: [Class] e.g. AutoModel, AutoModelForCausalLM
# @param device_map: [Dict/Str] default "auto"
# @param offload_folder: [Str] the folder path for unloading model
# @return model: Huggingface AutoModelForCausalLM object
def accelerate_load_model(model_path, 
						  Model = AutoModelForCausalLM,
						  device_map = "auto",
						  offload_folder = "./temp",
						  offload_buffers = True,
						  **kwargs
						  ):
	with init_empty_weights():
		model = AutoModelForCausalLM.from_pretrained(model_path)
	model = load_checkpoint_and_dispatch(
		model,
		checkpoint = model_path,
		device_map = device_map,
		offload_folder = offload_folder,
		offload_buffers = offload_buffers,
		dtype = torch.bfloat16,
		max_memory = {0: "75GB"},
		**kwargs,
	)
	return model


