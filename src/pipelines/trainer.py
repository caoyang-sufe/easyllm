# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@stu.sufe.edu.cn

import trl
import torch
import logging
from copy import deepcopy
from datasets import load_dataset
from transformers import (
	AutoConfig,
	AutoTokenizer,
	AutoModelForCausalLM,
	BitsAndBytesConfig,
	TrainingArguments,
	HfArgumentParser,
)
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model, PeftModel
from trl import (
	ScriptArguments, ModelConfig, 
	SFTConfig, SFTTrainer,
	PPOConfig, PPOTrainer,
	DPOConfig, DPOTrainer,
	GRPOConfig, GRPOTrainer,	# Use getattr instead
	get_peft_config, get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
from src.tools.trl import update_trl_config, generate_simple_data_processor
from src.modules import (
	ParallelQwen2Model, SkipLayerQwen2ForCausalLM, 
	ParallelQwen2ForCausalLM, SkipLayerQwen2ForCausalLM, 
	ParallelQwen3Model, SkipLayerQwen3ForCausalLM, 
	ParallelQwen3ForCausalLM, SkipLayerQwen3ForCausalLM, 
	ParallelLlamaModel, SkipLayerLlamaForCausalLM, 
	ParallelLlamaForCausalLM, SkipLayerLlamaForCausalLM, 
	SkipLayerDeepseekModel, SkipLayerDeepseekForCausalLM,
	ParallelDeepseekModel, ParallelDeepseekForCausalLM,
	SkipLayerDeepseekV2Model, SkipLayerDeepseekV2ForCausalLM,
	ParallelDeepseekV2Model, ParallelDeepseekV2ForCausalLM,
	SkipLayerDeepseekV3Model, SkipLayerDeepseekV3ForCausalLM,
	ParallelDeepseekV3Model, ParallelDeepseekV3ForCausalLM,
)

# Trainer Pipeline
# @param name: [Str] e.g. "SFT", "PPO", "DPO", "GRPO"
# @param train_data_processor: [Function] prepared for `train_dataset.map(data_processor)`, train_dataset is only one
# @param test_data_processors: List[Function] prepared for `test_dataset.map(data_processor)`, test_dataset may be more than one
# @param config_kwargs: [Dict] keyword arguments for updating TRL-Config, `ScriptArguments`, `ModelConfig`
#   - keyword arguments for `TRLConfig`: e.g. "output_dir", "adam_xxx", "learning_rate", "kl_coef", "push_to_hub"
#   - keyword arguments for `ScriptArguments`: e.g. "output_dir", "adam_xxx", "learning_rate", "kl_coef", "push_to_hub"
#   - keyword arguments for `ModelConfig`: e.g. "model_name_or_path", "torch_dtype", "trust_remote_code", "use_peft", "lora_xxx", "load_in_4bit", "bnb_4bit_compute_dtype", "bnb_4bit_quant_type"
# @param trainer_kwargs: [Dict] keyword arguments for updating TRL-Trainer
#   - keyword arguments for all Trainers: e.g. "data_collator", "callbacks", "train_dataset", "eval_dataset"
#   - keyword arguments for `SFTTrainer`: e.g. "compute_loss_func", "compute_metrics"
#   - keyword arguments for `PPOTrainer`: e.g. "ref_model[required]", "reward_model[required]", "value_model[required]"
#   - keyword arguments for `DPOTrainer`: e.g. "ref_model"
#   - keyword arguments for `GRPOTrainer`: e.g. "reward_funcs[required]"
# @param parallel_model_class: [Str] e.g. "ParallelQwen2ForCausalLM", "ParallelQwen2Model", default `None` refer to AutoModelForCausalLM
# @param n_cuda: [Int] Number of CUDA device available
# @param adapter_output_dirs: [List[Str]] All `output_dir` of adapters (usually trained by LoRA)
def base_pipeline(name, 
				  train_data_processor, 
				  test_data_processors,
				  config_kwargs, 
				  trainer_kwargs, 
				  parallel_model_class = None, 
				  n_cuda = 2,
				  adapter_output_dirs = None,
				  parse_arguments = False,
				  ):
	# ------------------------------------------------------------------
	# 1 Configuration
	TRLConfig, TRLTrainer = eval(f"{name}Config"), eval(f"{name}Trainer")
	if parse_arguments:
		parser = HfArgumentParser((ScriptArguments, TRLConfig, ModelConfig))
		script_arguments, trainer_config, model_config = parser.parse_args_into_dataclasses()
	else:
		script_arguments = ScriptArguments()
		trainer_config = TRLConfig()
		model_config = ModelConfig()
	script_arguments = update_trl_config(script_arguments, **config_kwargs)
	trainer_config = update_trl_config(trainer_config, **config_kwargs)
	model_config = update_trl_config(model_config, **config_kwargs)
	peft_config = get_peft_config(model_config)
	quantization_config = get_quantization_config(model_config)
	# ------------------------------------------------------------------
	# 2 Load models and tokenizer
	logging.info("Load models and tokenizer ...")
	logging.info(f"  - Model: {model_config.model_name_or_path}")
	tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
	if tokenizer.chat_template is None:
		tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
	if parallel_model_class is None:
		# 2.1 Model parallel settings
		logging.info("Using AutoModelForCausalLM ...")
		model = AutoModelForCausalLM.from_pretrained(
			model_config.model_name_or_path,
			device_map = "auto",
			trust_remote_code = model_config.trust_remote_code,
			quantization_config = quantization_config,
		)
	else:
		logging.info(f"Using {parallel_model_class} ...")
		model = eval(parallel_model_class).from_pretrained(
			model_config.model_name_or_path,
			n_cuda = n_cuda,
			device_map = "cpu",
		)
		model.module_to_device()
	if adapter_output_dirs is not None:
		# 2.2 Load adapters
		logging.info(f"  - Load adapters ...")
		for i, adapter_output_dir in enumerate(adapter_output_dirs):
			logging.info(f"    - Load adapter {i}: {adapter_output_dir}")
			model = PeftModel.from_pretrained(model, model_id = adapter_output_dir)
			model = model.merge_and_unload()
	logging.info(f"Print parameter device ...")
	for name, parameter in model.named_parameters():
		logging.info(f"{name}: {parameter.device}")
	if peft_config is not None:
		# 2.3 Model PEFT settings
		logging.info("Prepare model for PEFT ...")
		model.config.pretraining_tp = 1
		model.config.use_cache = False
		model.gradient_checkpointing_enable()
		# If `prepare_model_for_kbit_training` is ignored, and `gradient_checkpointing = True` (for GPU memory saving)
		# Then you need set `model.enable_input_require_grads()` yourself
		# model = prepare_model_for_kbit_training(model)
		model.enable_input_require_grads()
		model = get_peft_model(model, peft_config)
	if name == "PPO":
		# 2.4 PPO is special! It needs more components!
		logging.info("PPO load reward value and reference models ...")
		logging.info(f"  - Reward model: {trainer_config.reward_model_path}")
		reward_model = AutoModelForSequenceClassification.from_pretrained(
			trainer_config.reward_model_path,
			trust_remote_code = model_config.trust_remote_code,
			num_labels = 1,
		)
		value_model = AutoModelForSequenceClassification.from_pretrained(
			trainer_config.reward_model_path,
			trust_remote_code = model_config.trust_remote_code,
			num_labels = 1,
		)
		reward_tokenizer = AutoTokenizer.from_pretrained(trainer_config.reward_model_path)
		if reward_tokenizer.chat_template is None:
			reward_tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
		logging.info("  - Copy reference model ...")
		# Clone model: I prefer deepcopy
		ref_model = deepcopy(model)
		# ref_model = model.__class__(model.config)
		# ref_model.load_state_dict(model.state_dict())
		trainer_kwargs["reward_model"] = reward_model
		trainer_kwargs["value_model"] = value_model
		trainer_kwargs["ref_model"] = ref_model
		trainer_kwargs["processing_class"] = reward_tokenizer
		logging.info("  - Done!")
		if train_data_processor is None:
			# The data processor of PPO is also different to others
			def train_data_processor(_data):
				outputs = tokenizer(_data["prompt"] + _data["completion"], padding = False)
				return {"input_ids": outputs["input_ids"]}
	else:
		trainer_kwargs["processing_class"] = tokenizer
	# ------------------------------------------------------------------
	# 3 Load dataset
	logging.info("Load dataset ...")
	logging.info(f"  - Dataset: {script_arguments.dataset_name}")
	if train_data_processor is None:
		train_data_processor = generate_simple_data_processor(name)	# Default data processor
	if not "train_dataset" in trainer_kwargs:
		# 3.1 Train dataset
		if isinstance(script_arguments.dataset_name, str):
			# Train and test splits are from the same dataset
			train_dataset_path = script_arguments.dataset_name
		elif isinstance(script_arguments.dataset_name, dict):
			# Recommend: more robust usage: {"train": <dataset_name>}
			train_dataset_path = script_arguments.dataset_name["train"]			
		else:
			raise Exception(f"Unexpected script argument `dataset_name`: {script_arguments.dataset_name}")	
		train_dataset = load_dataset(train_dataset_path, split=script_arguments.dataset_train_split)
		train_dataset = train_dataset.map(train_data_processor, remove_columns=train_dataset.column_names)
		logging.info(f"  - Train dataset: {len(train_dataset)}")
		trainer_kwargs["train_dataset"] = train_dataset
	if not "eval_dataset" in trainer_kwargs:
		# 3.2 Evaluation dataset(s)
		if isinstance(script_arguments.dataset_test_split, str):
			test_splits = [script_arguments.dataset_test_split]
		if isinstance(script_arguments.dataset_test_split, list):
			test_splits = script_arguments.dataset_test_split[:]
		else:
			raise Exception(f"Unexpected script argument `dataset_test_split`: {script_arguments.dataset_test_split}")
		if test_data_processors is None:
			test_data_processors = [train_data_processor] * len(test_splits)
		assert len(test_splits) == len(test_data_processors), f"Mismatch: {test_splits} and {test_data_processors}"
		n_test_datasets = len(test_splits)
		if isinstance(script_arguments.dataset_name, str):
			test_dataset_paths = [script_arguments.dataset_name] * n_test_datasets
		elif isinstance(script_arguments.dataset_name, dict):
			# Recommend: more robust usage: {"test": List[<dataset_name>]}
			if isinstance(script_arguments.dataset_name["test"], str):
				test_dataset_paths = [script_arguments.dataset_name["test"]] * n_test_datasets
			elif isinstance(script_arguments.dataset_name["test"], list):
				test_dataset_paths = script_arguments.dataset_name["test"][:]
			else:
				raise Exception(f"Unexpected script argument `dataset_name`: {script_arguments.dataset_name}")
		else:
			raise Exception(f"Unexpected script argument `dataset_name`: {script_arguments.dataset_name}")
		eval_dataset = dict()
		for i in range(n_test_datasets):
			eval_dataset[f"eval_{i}"] = load_dataset(test_dataset_paths[i], split=test_splits[i])
			eval_dataset[f"eval_{i}"] = eval_dataset[f"eval_{i}"].map(test_data_processors[i], remove_columns=eval_dataset[f"eval_{i}"].column_names)
		logging.info(f"  - {n_test_datasets} Eval dataset: {[len(dataset) for dataset in eval_dataset]}")			
		trainer_kwargs["eval_dataset"] = eval_dataset
	# ------------------------------------------------------------------
	# 4 Train model
	logging.info("Trainer starts ...")
	trainer = TRLTrainer(
		model = model,
		args = trainer_config,
		peft_config = peft_config,
		**trainer_kwargs	# train_dataset, eval_dataset, processing_class
	)
	trainer.train()
	logging.info("  - Trainer finishes!")
	# ------------------------------------------------------------------
	# 5 Save model
	if trainer_config.push_to_hub:
		# 5.1 Push to HuggingFace Hub
		logging.info(f"  - Push checkpoints to {trainer_config.organization}/{trainer_config.push_to_hub_model_id}")
		trainer.push_to_hub()
		NotImplemented
	logging.info(f"Save model to {trainer_config.output_dir}")
	trainer.save_model(trainer_config.output_dir)

# SFT Pipeline
def sft_pipeline(train_data_processor, test_data_processors, config_kwargs, trainer_kwargs, parallel_model_class = None, n_cuda = 2, adapter_output_dirs = None, parse_arguments = False):
	base_pipeline(
		"SFT", 
		train_data_processor, 
		test_data_processors,
		config_kwargs, 
		trainer_kwargs, 
		parallel_model_class, 
		n_cuda,
		adapter_output_dirs,
		parse_arguments,
	)

# PPO Pipeline
def ppo_pipeline(train_data_processor, test_data_processors, config_kwargs, trainer_kwargs, parallel_model_class = None, n_cuda = 2, adapter_output_dirs = None, parse_arguments = False):
	base_pipeline(
		"PPO", 
		train_data_processor, 
		test_data_processors,
		config_kwargs, 
		trainer_kwargs, 
		parallel_model_class, 
		n_cuda,
		adapter_output_dirs,
		parse_arguments,
	)

# DPO Pipeline
def dpo_pipeline(train_data_processor, test_data_processors, config_kwargs, trainer_kwargs, parallel_model_class = None, n_cuda = 2, adapter_output_dirs = None, parse_arguments = False):
	base_pipeline(
		"DPO", 
		train_data_processor, 
		test_data_processors,
		config_kwargs, 
		trainer_kwargs, 
		parallel_model_class, 
		n_cuda,
		adapter_output_dirs,
		parse_arguments,
	)

# GRPO Pipeline
def grpo_pipeline(train_data_processor, test_data_processors, config_kwargs, trainer_kwargs, parallel_model_class = None, n_cuda = 2, adapter_output_dirs = None, parse_arguments = False):
	base_pipeline(
		"GRPO", 
		train_data_processor, 
		test_data_processors,
		config_kwargs, 
		trainer_kwargs, 
		parallel_model_class, 
		n_cuda,
		adapter_output_dirs,
		parse_arguments,
	)
