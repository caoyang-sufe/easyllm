# -*- coding: utf8 -*-
# @author: caoyang
# @email: caoyang@stu.sufe.edu.cn

import os
import pandas
import logging

from src.unittests import model_home, dataset_home, model_names, dataset_names
from src.pipelines.generate import decode_pipeline, generate_pipeline

def decode_pipeline_test():
	logging.info("Decode unittest ...")
	model_id = 0
	model_name_or_path = os.path.join(model_home, model_names[model_id])
	logging.info(f"  - Model: {model_name_or_path}")
	prompt = """英文单词strawberry中有几个字母r？</think>"""
	max_length = 32
	use_kv_cache = False
	df_display = decode_pipeline(model_name_or_path,
								 prompt,
								 max_length,
								 device = None,
								 use_kv_cache = use_kv_cache,
								 )
	save_path = f"./decode+{model_names[model_id].split('/')[-1]}+{use_kv_cache}.csv"
	logging.info(f"Export to {save_path}")
	df_display.to_csv(save_path, sep='\t', header=True, index=False)
	logging.info("  - OK!")

def generate_pipeline_test():
	logging.info("Generate unittest ...")
	model_id = 0
	model_name_or_path = os.path.join(model_home, model_names[model_id])
	logging.info(f"  - Model: {model_name_or_path}")
	prompt = """英文单词strawberry中有几个字母r？</think>"""
	max_length = 40
	df_display = generate_pipeline(model_name_or_path,
								   prompt,
								   max_length,
								   device = None,
								   generate_kwargs = None,
								   )
	save_path = f"./generate+{model_names[model_id].split('/')[-1]}.csv"
	logging.info(f"Export to {save_path}")
	df_display.to_csv(save_path, sep='\t', header=True, index=False)
	logging.info("  - OK!")
