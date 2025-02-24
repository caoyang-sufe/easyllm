# -*- coding: utf8 -*-
# @author: caoyang
# @email: caoyang@stu.sufe.edu.cn


def demo_1(mid=0):
	from transformers import AutoTokenizer, AutoModelForCausalLM
	import os
	import json
	import torch
	from src.tools.transformers import greedy_decode, beam_search_decode	

	model_paths = [
		"/data/nishome/wangyinglin/caoyang/ds-test/DeepSeek-R1-Distill-Qwen-1.5B",
		"/data/nishome/wangyinglin/yangyitong/DeepSeek-R1-Distill-Qwen-32B",
	]
	model_path = model_paths[mid]
	
	input_text = "很久很久以前，"
	max_length = 64
	device = "cuda"
	tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
	model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).to(device)
	
	generated_text = greedy_decode(
		model,
		tokenizer,
		input_text, 
		max_length,
		device,
	)
	with open("./greedy_decode.txt", 'w', encoding="utf8") as f:
		f.write(generated_text)

	n_branches = 3
	depth = 2
	generated_text = beam_search_decode(
		model,
		tokenizer,
		input_text,
		max_length,
		n_branches,
		depth,
		device,
	)
	with open("./beam_search_decode.txt", 'w', encoding="utf8") as f:
		f.write(generated_text)
# 2 1 2
# 2 2 4
# 3 2 9
demo_1(0)
	