# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@stu.sufe.edu.cn

from copy import deepcopy
from transformers import HfArgumentParser, TrainingArguments
from trl import PPOConfig, SFTConfig, ModelConfig, ScriptArguments

# @param config: TRLConfig object
# @param kwargs: [Dict] updated keyword arguments
def update_trl_config(config, **kwargs):
	config_dataclass_fields = list(config.__dataclass_fields__.keys())
	for key, value in kwargs.items():
		if key in config_dataclass_fields:
			config.__setattr__(key, value)
	return config

def generate_trl_config(*Config, **kwargs):
	parser, = HfArgumentParser(*Config)
	config = parser.parse_args_into_dataclasses()
	for key, value in kwargs.items():
		config.__setattr__(key, value)
	return config

# @param name: [Str] e.g. "SFT", "PPO", "DPO", "GRPO"
def generate_simple_data_processor(name, **kwargs):
	if name in ["SFT", "GRPO"]:
		def _data_processor(_data):
			return {"prompt": _data["prompt"], "completion": _data["completion"]}
	elif name == "PPO":
		tokenizer = kwargs.get("tokenizer")
		def _data_processor(_data):
			outputs = tokenizer(_data["prompt"] + _data["completion"], padding = False)
			return {"input_ids": outputs["input_ids"]}
	elif name == "DPO":
		def _data_processor(_data):
			return {"prompt": _data["prompt"], "chosen": _data["chosen"], "rejected": _data["rejected"]}
	else:
		raise NotImplementedError(name)
	return _data_processor
