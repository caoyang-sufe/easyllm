# -*- coding: utf8 -*-
# @author: caoyang
# @email: caoyang@stu.sufe.edu.cn
# Evaluator for CAUSAL_LM

import os
import json
import time
import logging
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from src.unittests import model_home, dataset_home, model_names, dataset_names
from src.pipelines.evaluator import base_pipeline
from src.module import (
	ParallelQwen2Model, SkipLayerQwen2ForCausalLM, 
	ParallelQwen2ForCausalLM, SkipLayerQwen2ForCausalLM, 
	ParallelQwen3Model, SkipLayerQwen3ForCausalLM, 
	ParallelQwen3ForCausalLM, SkipLayerQwen3ForCausalLM, 
	ParallelLlamaModel, SkipLayerLlamaForCausalLM, 
	ParallelLlamaForCausalLM, SkipLayerLlamaForCausalLM, 
)

def evaluate_causal_lm_test(model_id=10, parallel_model_class=None, n_cuda=2):
	model_name_or_path = os.path.join(model_home, model_names[model_id])
	tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
	if parallel_model_class is None:
		logging.info("Using AutoModelForCausalLM ...")
		model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
	else:
		logging.info(f"Using {parallel_model_class} ...")
		model = eval(parallel_model_class).from_pretrained(model_name_or_path, n_cuda = n_cuda)
	dataset_path = os.path.join(dataset_home, dataset_names[4])
	base_pipeline(
		model = model,
		tokenizer = tokenizer,
		dataset = None,
		model_name_or_path = None,
		dataset_name_or_path = dataset_path,
		test_data_split = "test",
		test_data_size = 500,
		device = "cpu",
		input_column = "prompt",
		target_column = "completion",
		do_sample = False,
		do_sample_times = 10,
		do_sample_kwargs = {"top_k": 0, "top_p": 1., "temperature": 1., "num_beams": 1},
		use_cache = True,
		input_max_length = 512,
		target_max_length = 128,
	)
