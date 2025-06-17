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
	model_name_or_path = os.path.join(model_home, model_names[0])
	logging.info(f"  - Model: {model_name_or_path}")
	prompt = """英文单词strawberry中有几个字母r？"""
	max_length = 128
	df_display = decode_pipeline(model_name_or_path,
								 prompt,
								 max_length,
								 device = None,
								 use_kv_cache = True,
								 )
	df_display.to_csv("./decode_test.csv", sep='\t', header=True, index=False)
