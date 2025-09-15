# -*- coding: utf8 -*-
# @author: caoyang
# @email: caoyang@stu.sufe.edu.cn

import re
import torch
import logging
from torch.nn import functional as F
from matplotlib import pyplot as plt
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

from src.tools.plot import plot_tensor_heatmap
from src.tools.transformers import greedy_decode
from src.module.skip_layer import create_model_class, create_model_class_for_causal_lm
from src.pipelines.generate import display_pipeline

# Horizontal comparison: Compare hook data (which comes from different prompts) by module names
# Focusing on comparing the inputs or outputs of the same modules in different hooks
# @param hook_datas: List[Dict] of length 2, i.e. currently we only compare two series of hook data 
# @param hook_data_paths: List[Str], default None but at least one of `hook_datas` and `hook_data_paths` is not None
# @param hook_module_names: List[Str], e.g. ["model.layers[0].self_attn.q_proj", "model.layers[0].self_attn.k_proj", "model.layers[0].self_attn.v_proj"]
# @param hook_module_name_suffixes: List[Str], e.g. ["q_proj", "k_proj", "v_proj"]
# @param comparison_index: List[Str], e.g. ["mean_diff", "max_diff", "corr"]
# @param max_length: [Int] when generating one token, one comparison is conducted. So we need to limit the max comparison by `max_length`
# @param figure_size: [Int] default 5
def horizontal_comparison_of_forward_hook(
	hook_datas = None,
	hook_data_paths = None,
	hook_module_names = ["model.layers[0].self_attn.q_proj", "model.layers[1].self_attn.k_proj", "model.layers[2].self_attn.v_proj",],
	hook_module_name_suffixes = ["q_proj", "k_proj", "v_proj"],
	comparison_index = ["mean_diff", "max_diff", "corr", "sim"],
	max_length = 999,
	figure_size = 5,
):
	regex = re.compile("\[\d+\]", re.I)	# e.g. Used to match `[0]` in `model.layers[0].self_attn.q_proj`
	assert hook_datas is not None or hook_data_paths is not None
	if hook_datas is None:
		hook_datas = [torch.load(hook_data_path) for hook_data_path in hook_data_paths]
	for token_i, hook_data_group in enumerate(zip(*hook_datas)):
		print(token_i)
		if token_i >= max_length:
			break
		# Hook data when generating token i
		# Summary dictionary contains all the compared
		comparison_summary_dict = {
			"mean_diff": {module_name_suffix: {"input": list(), "output": list()} for module_name_suffix in hook_module_name_suffixes},	# Mean difference between two tensors
			"max_diff": {module_name_suffix: {"input": list(), "output": list()} for module_name_suffix in hook_module_name_suffixes},	# Max differernce by element-wise value between two tensors
			"corr": {module_name_suffix: {"input": list(), "output": list()} for module_name_suffix in hook_module_name_suffixes},	# Correlation coefficient between two tensors
			"sim": {module_name_suffix: {"input": list(), "output": list()} for module_name_suffix in hook_module_name_suffixes},	# Similarity between two tensors
		}
		for module_name in hook_module_names:
			module_name_suffix = module_name.split('.')[-1]
			module_name_suffix = regex.sub(str(), module_name_suffix)
			if module_name_suffix in hook_module_name_suffixes:
				# 1. Process inputs in hook data
				input_data_group = [data[module_name].get("input", data[module_name].get("args")) for data in hook_data_group]
				for j, input_data in enumerate(input_data_group):     
					# Assertation for ensuring data format of inputs
					assert len(input_data) == 1 and isinstance(input_data[0], tuple)
					if len(input_data[0]) > 1:
						logging.warning(f"Input data {j} has more than 1 components: {len(input_data[0])}")
				input_tensors = [input_data[0][0].float() for input_data in input_data_group]
				for j, input_tensor in enumerate(input_tensors):
					logging.info(f"Size of input tensor {j}: {input_tensor.size()}")
				# 2. Process outputs in hook data
				output_data_group = [data[module_name]["output"] for data in hook_data_group]
				output_tensors = list()
				for j, output_data in enumerate(output_data_group):
					# Assertation for ensuring data format of outputs
					assert len(output_data) == 1
					if isinstance(output_data[0], torch.Tensor):
						output_tensor = output_data[0]
					else:
						assert isinstance(output_data[0], tuple)
						if len(output_data[0]) > 1:
							logging.warning(f"Output data {j} has more than 1 components: {len(output_data[0])}")
						output_tensor = output_tensor[0][0]
					output_tensors.append(output_tensor)
					
				# 3. Summary data calculation
				## 3.1 Calculate Mean Difference
				input_diff = input_tensors[0] - input_tensors[1]
				output_diff = output_tensors[0] - output_tensors[1]
				mean_input_diff = torch.norm(input_diff, p="fro") / input_tensors[0].numel()
				mean_output_diff = torch.norm(output_diff, p="fro") / output_tensors[0].numel()
				comparison_summary_dict["mean_diff"][module_name_suffix]["input"].append(mean_input_diff.item())
				comparison_summary_dict["mean_diff"][module_name_suffix]["output"].append(mean_output_diff.item())
				## 3.2 Calculate Max Difference
				max_input_diff = torch.max(torch.abs(input_diff))
				max_output_diff = torch.max(torch.abs(output_diff))
				comparison_summary_dict["max_diff"][module_name_suffix]["input"].append(max_input_diff.item())
				comparison_summary_dict["max_diff"][module_name_suffix]["output"].append(max_output_diff.item())
				## 3.3 Calculate Correlation Coefficient
				input_corr = torch.corrcoef(torch.stack([input_tensors[0].flatten(), input_tensors[1].flatten()]))[0, 1]
				output_corr = torch.corrcoef(torch.stack([output_tensors[0].flatten(), output_tensors[1].flatten()]))[0, 1]
				comparison_summary_dict["corr"][module_name_suffix]["input"].append(input_corr.item())
				comparison_summary_dict["corr"][module_name_suffix]["output"].append(output_corr.item())
				## 3.4 Calculate Similarity
				input_sim = F.cosine_similarity(input_tensors[0].flatten(), input_tensors[1].flatten(), dim=0)
				output_sim = F.cosine_similarity(output_tensors[0].flatten(), output_tensors[1].flatten(), dim=0)
				comparison_summary_dict["sim"][module_name_suffix]["input"].append(input_sim.item())
				comparison_summary_dict["sim"][module_name_suffix]["output"].append(output_sim.item())				
				## 3.5 TO BE CONTINUE ......
				# ...
		nrows, ncols = len(comparison_index), len(hook_module_name_suffixes)
		fig, axes = plt.subplots(
			nrows = nrows, 
			ncols = ncols,
			figsize = (figure_size * 1.2 * ncols, figure_size * nrows),
		)
		for i, summary_key in enumerate(comparison_index):
			for j, module_name_suffix in enumerate(hook_module_name_suffixes):
				y_input = comparison_summary_dict[summary_key][module_name_suffix]["input"]
				y_output = comparison_summary_dict[summary_key][module_name_suffix]["output"]
				assert len(y_input) == len(y_output)
				x = range(len(y_input))
				if len(x) == 0:
					# No inputs exist
					continue
				if len(comparison_index) == 1 and len(hook_module_name_suffixes) == 1:
					target_ax = axes
				elif len(comparison_index) == 1:
					target_ax = axes[j]
				elif len(hook_module_name_suffixes) == 1:
					target_ax = axes[i]
				else:
					target_ax = axes[i, j]
				target_ax.bar(x, y_input, label=f"input_{summary_key}", alpha=.5)
				target_ax.bar(x, y_output, label=f"output_{summary_key}", alpha=.5)
				target_ax.set_xlabel("Layer #"), target_ax.set_ylabel(summary_key), target_ax.set_title(f"{summary_key} for {module_name_suffix} on token {token_i}")
				target_ax.legend()
		plt.show(), plt.close()
	
# Vertical comparison: Compare data in a single hook
# Focusing on comparing the inputs and outputs of the same modules
# @param hook_data: [Dict] hook data object
# @param hook_data_path: [Str] default None but at least one of `hook_datas` and `hook_data_paths` is not None
# @param hook_module_names: List[Str], e.g. ["model.layers[0]"]
# @param comparison_index: List[Str], e.g. ["mean_diff", "max_diff", "corr"]
# @param max_length: [Int] when generating one token, one comparison is conducted. So we need to limit the max comparison by `max_length`
# @param figure_size: [Int] default 5
# @param watched_module_names: List[Int], you can selected several module here to plot heat map of input-output difference
def vertical_comparison_of_forward_hook(
	hook_data = None,
	hook_data_path = None,
	hook_module_names = ["model.layers[0]", "model.layers[1]", "model.layers[2]"],
	comparison_index = ["mean_diff", "max_diff", "corr", "sim"],
	max_length = 999,
	figure_size = 5,
	watched_module_names = ["model.layers[0]"],
):
	assert hook_data is not None or hook_data_path is not None
	if hook_data is None:
		hook_data = torch.load(hook_data_path, weights_only=False)
	for token_i in range(max_length):
		comparison_summary_dict = {
			"mean_diff": list(),
			"max_diff": list(),
			"corr": list(),
			"sim": list()
		}
		# Plot heatmap of input-output difference
		fig, axes = plt.subplots(1, len(watched_module_names), figsize=(1.2 * 5 * figure_size * len(watched_module_names), figure_size))
		subplot_index = -1
		for module_name in hook_module_names:
			input_tensor = hook_data[token_i][module_name].get("input", hook_data[token_i][module_name].get("args"))[0][0]
			output_tensor = hook_data[token_i][module_name]["output"][0][0]
			diff = input_tensor - output_tensor
			mean_diff = torch.norm(diff, p="fro") / input_tensor.numel()                                                              
			max_diff = torch.max(torch.abs(diff))
			corr = torch.corrcoef(torch.stack([input_tensor.flatten(), output_tensor.flatten()]))[0, 1]
			sim = F.cosine_similarity(input_tensor.flatten(), output_tensor.flatten(), dim=0)
			comparison_summary_dict["mean_diff"].append(mean_diff.item())
			comparison_summary_dict["max_diff"].append(max_diff.item())
			comparison_summary_dict["corr"].append(corr.item())
			comparison_summary_dict["sim"].append(sim.item())
			if module_name in watched_module_names:
				subplot_index += 1
				assert diff.size(0) == 1
				plot_tensor_heatmap(
					tensor = torch.abs(diff)[0, :, :],
					ax = axes[subplot_index] if len(watched_module_names) > 1 else axes, 
					is_show=False, 
					title=f"Diff in {module_name} of Token {token_i}",
				)
		plt.show(), plt.close()
		# Plot line chart of comparison index
		ncols = len(comparison_index)
		nrows = 1
		fig, axes = plt.subplots(
			nrows = nrows, 
			ncols = ncols, 
			figsize = (ncols * figure_size * 1.2, nrows * figure_size),
		)
		for c, summary_key in enumerate(comparison_index):
			for r in range(nrows):
				if ncols == 1 and nrows == 1:
					target_ax = axes
				elif ncols == 1:
					target_ax = axes[r]
				elif nrows == 1:
					target_ax = axes[c]
				else:
					target_ax = axes[r, c]
				x = list(range(len(hook_module_names)))
				target_ax.plot(x, comparison_summary_dict[summary_key], label=summary_key, marker='o')
				# Plot text on each dot
				if summary_key in ["corr", "sim"]:
					for i, (x_i, y_i) in enumerate(zip(x, comparison_summary_dict[summary_key])):
						text_flag = i % 3 == 0
						last_y_flag = abs(y_i - comparison_summary_dict[summary_key][i - 1]) > .1 if i > 0 else True
						next_y_flag = abs(y_i - comparison_summary_dict[summary_key][i + 1]) > .1 if i < len(x) - 1 else True
						if text_flag or (last_y_flag and next_y_flag):
							target_ax.text(x_i, y_i, str(round(y_i, 3)), ha="center", va="bottom", fontsize=12, color="red")
					
				target_ax.legend(), target_ax.set_xlabel("Layer #"), target_ax.set_ylabel(summary_key), target_ax.set_title(f"{comparison_index[c]} on token {token_i}")
		plt.show(), plt.close()

# Generating by skipping decoder blocks
# Focusing on the generating results under skipping different layers
# @param Model: AutoModel class, e.g. Qwen2Model
# @param ModelForCausalLM: AutoModelForCausalLM class, e.g. Qwen2ForCausalLM
# @param model_name_or_path: Str
# @param tokenizer: HuggingFace tokenizer object
# @param prompt: Str
# @param max_length: Int
# @param skip_layer_ids: List[Int], Layer # to be skipped
def skip_layer_generation(
	Model, 
	ModelForCausalLM,
	model_name_or_path,
	tokenizer,
	prompt, 
	max_length,
	skip_layer_ids = list(),
):
	SkipLayerModelForCausalLM = create_model_class_for_causal_lm(Model, ModelForCausalLM)
	config = AutoConfig.from_pretrained(model_name_or_path)
	model = SkipLayerModelForCausalLM.from_pretrained(
		model_name_or_path,
		config = config,
		skip_layer_ids = skip_layer_ids,
	)
	results = greedy_decode(
		model,
		tokenizer,
		prompt = prompt, 
		max_length = max_length,
		device = "cpu",
		use_kv_cache = False,
		forward_hook_module_names = None,
		backward_hook_module_names = None,
	)
	return results

# Generating by skipping decoder blocks
# Focusing on the generating results under skipping different layers
# @param model: HuggingFace model object
# @param tokenizer: HuggingFace tokenizer object
# @param prompt: Str
# @param max_length: Int
# @param skip_layer_ids: List[Int], Layer # to be skipped
def easy_skip_layer_generation(
	model,
	tokenizer,
	prompt, 
	max_length,
	skip_layer_ids = list(),
	use_kv_cache = True,
	forward_hook_module_names = None,
	backward_hook_module_names = None,
):
	backup_layers = model.model.layers
	if skip_layer_ids:
		filtered_layers = [
			layer for i, layer in enumerate(model.model.layers)
			if i not in skip_layer_ids
		]
		model.model.layers = torch.nn.ModuleList(filtered_layers)
	results = greedy_decode(
		model,
		tokenizer,
		prompt = prompt, 
		max_length = max_length,
		device = "cpu",
		use_kv_cache = use_kv_cache,
		forward_hook_module_names = forward_hook_module_names,
		backward_hook_module_names = backward_hook_module_names,
	)
	model.model.layers = backup_layers
	return results

	
