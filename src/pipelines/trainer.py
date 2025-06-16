# -*- coding: utf8 -*-
# @author: caoyang
# @email: caoyang@stu.sufe.edu.cn

import wandb
import logging
from copy import deepcopy
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, HfArgumentParser
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
from trl import (
	ScriptArguments, ModelConfig, 
	SFTConfig, SFTTrainer,
	PPOConfig, PPOTrainer,
	DPOConfig, DPOTrainer,
	GRPOConfig, GRPOTrainer,
	get_peft_config, get_quantization_config,
)
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
from src.tools.trl import update_trl_config, generate_simple_data_processor

# Trainer Pipeline
# @param name: [Str] e.g. "SFT", "PPO", "DPO", "GRPO"
# @param data_processor: Function object prepared for `dataset.map(data_processor)`
# @param trainer_config: [Dict, peft.XXXConfig] including keyword arguments, e.g. 
# @param model_config: [Dict, peft.ModelConfig] including keyword arguments, e.g. 
# @param script_arguments: [Dict, peft.ScriptArguments] including keyword arguments, e.g. "dataset_name", "dataset_train_split", "dataset_test_split"
# @param config_kwargs: [Dict] keyword arguments for updating TRL-Config, `ScriptArguments`, `ModelConfig`
#   - keyword arguments for `TRLConfig`: e.g. "output_dir", "adam_xxx", "learning_rate", "kl_coef", "push_to_hub"
#   - keyword arguments for `ScriptArguments`: e.g. "output_dir", "adam_xxx", "learning_rate", "kl_coef", "push_to_hub"
#   - keyword arguments for `ModelConfig`: e.g. "model_name_or_path", "torch_dtype", "trust_remote_code", "use_peft", "lora_xxx", "load_in_4bit", "bnb_4bit_compute_dtype", "bnb_4bit_quant_type"
# @param trainer_kwargs: [Dict] keyword arguments for updating TRL-Trainer
#   - keyword arguments for all Trainers: e.g. "data_collator", "callbacks"
#   - keyword arguments for `SFTTrainer`: e.g. "compute_loss_func", "compute_metrics"
#   - keyword arguments for `PPOTrainer`: e.g. "ref_model[required]", "reward_model[required]", "value_model[required]"
#   - keyword arguments for `DPOTrainer`: e.g. "ref_model"
#   - keyword arguments for `GRPOTrainer`: e.g. "reward_funcs[required]"
def base_pipeline(name, data_processor, config_kwargs, trainer_kwargs):
	# 1 Configuration
	TRLConfig, TRLTrainer = eval(f"{name}Config"), eval(f"{name}Trainer")
	parser = HfArgumentParser((ScriptArguments, TRLConfig, ModelConfig))
	script_arguments, trainer_config, model_config = parser.parse_args_into_dataclasses()
	script_arguments = update_trl_config(script_arguments, **config_kwargs)
	trainer_config = update_trl_config(trainer_config, **config_kwargs)
	model_config = update_trl_config(model_config, **config_kwargs)
	peft_config = get_peft_config(model_config)
	quantization_config = get_quantization_config(model_config)
	# 2 Load models and tokenizer
	logging.info("Load models and tokenizer ...")
	logging.info(f"  - Model: {model_config.model_name_or_path}")
	tokenizer = AutoTokenizer.from_pretrained(model_config.model_name_or_path)
	if not "pad_token" in tokenizer.special_tokens_map:
		tokenizer.add_special_tokens({"pad_token": "[PAD]"})
	if tokenizer.chat_template is None:
		tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
	model = AutoModelForCausalLM.from_pretrained(
		model_config.model_name_or_path,
		device_map = "auto",
		trust_remote_code = model_config.trust_remote_code,
		quantization_config = quantization_config,
	)
	if peft_config is not None:
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
		logging.info("PPO load reward value and reference models ...")
		# PPO is special! It needs more components!
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
		if not "pad_token" in reward_tokenizer.special_tokens_map:
			reward_tokenizer.add_special_tokens({"pad_token": "[PAD]"})
		if reward_tokenizer.chat_template is None:
			reward_tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
		logging.info("  - Copy reference model ...")
		ref_model = deepcopy(model)
		# ref_model = model.__class__(model.config)
		# ref_model.load_state_dict(model.state_dict())
		trainer_kwargs["reward_model"] = reward_model
		trainer_kwargs["value_model"] = value_model
		trainer_kwargs["ref_model"] = ref_model
		trainer_kwargs["processing_class"] = reward_tokenizer
		logging.info("  - Done!")
		if data_processor is None:
			# The data processor of PPO is also different to others
			def data_processor(_data):
				outputs = tokenizer(_data["prompt"] + _data["completion"], padding = False)
				return {"input_ids": outputs["input_ids"]}
	else:
		trainer_kwargs["processing_class"] = tokenizer

	# 2 Load dataset
	logging.info("Load dataset ...")
	logging.info(f"  - Dataset: {script_arguments.dataset_name}")
	if data_processor is None:
		data_processor = generate_simple_data_processor(name)
	train_dataset = load_dataset(script_arguments.dataset_name, split=script_arguments.dataset_train_split)
	eval_dataset = load_dataset(script_arguments.dataset_name, split=script_arguments.dataset_test_split)
	train_dataset = train_dataset.map(data_processor, remove_columns=train_dataset.column_names)
	eval_dataset = eval_dataset.map(data_processor, remove_columns=eval_dataset.column_names)
	logging.info(f"  - Train dataset: {len(train_dataset)}")
	logging.info(f"  - Eval dataset: {len(eval_dataset)}")
	# 4 Train model
	logging.info("Trainer starts ...")
	trainer = TRLTrainer(
		model = model,
		args = trainer_config,
		train_dataset = train_dataset,
		eval_dataset = eval_dataset,
		peft_config = peft_config,
		**trainer_kwargs
	)
	trainer.train()
	logging.info("  - Trainer finishes!")
	# 5 Save model
	if trainer_config.push_to_hub:
		logging.info(f"  - Push checkpoints to {trainer_config.organization}/{trainer_config.push_to_hub_model_id}")
		trainer.push_to_hub()
	logging.info(f"Save model to {trainer_config.output_dir}")
	trainer.save_model(trainer_config.output_dir)

# SFT Pipeline
def sft_pipeline(data_processor, config_kwargs, trainer_kwargs):
	base_pipeline(
		name = "SFT",
		data_processor = data_processor,
		config_kwargs = config_kwargs,
		trainer_kwargs = trainer_kwargs,
	)

# PPO Pipeline
def ppo_pipeline(data_processor, config_kwargs, trainer_kwargs):
	base_pipeline(
		name = "PPO",
		data_processor = data_processor,
		config_kwargs = config_kwargs,
		trainer_kwargs = trainer_kwargs,
	)

# DPO Pipeline
def dpo_pipeline(data_processor, config_kwargs, trainer_kwargs):
	base_pipeline(
		name = "DPO",
		data_processor = data_processor,
		config_kwargs = config_kwargs,
		trainer_kwargs = trainer_kwargs,
	)

# GRPO Pipeline
def grpo_pipeline(data_processor, config_kwargs, trainer_kwargs):
	base_pipeline(
		name = "GRPO",
		data_processor = data_processor,
		config_kwargs = config_kwargs,
		trainer_kwargs = trainer_kwargs,
	)