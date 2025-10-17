# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@stu.sufe.edu.cn

import json
import torch
import numpy
import seaborn as sns
from safetensors import safe_open
from matplotlib import pyplot as plt

# Plot mean and variance of a sequence of tensors
# @param tensors: List[torch.Tensor]
# @param ax: Matplotlib subplot object
# @param figsize: [Tuple[Int, Int]] Only take effect when `ax` is None
# @param save_path: [Str] Figure save path
# @param is_show: [Boolean] Whether to show figure
# @param x_label: [Str]
# @param y_labels: [Tuple[Str, Str]] Note that here are two Y axises
# @param title: [Str] Figure title
# @param colors: [Tuple[Str, Str]] Two colors for mean and variance curves
# @return means: [List[Float]] of length `len(tensors)`
# @param variances: [List[Float]] of length `len(tensors)`
def plot_tensor_mean_and_variance(tensors,
								  ax = None,
								  figsize=(10, 8),
						  		  save_path = None,
						  		  is_show = True,
						  		  x_label = "Tensors",
						  		  y_labels = ("Mean", "Variance"),
						  		  title = "Trend of Mean and Variance",
						  		  colors = ("red", "blue"),
						  		  ):
	x = list(range(len(tensors)))
	means = [tensor.mean().item() for tensor in tensors]
	variances = [tensor.var().item() for tensor in tensors]
	color_mean, color_variance = colors
	if ax is None:
		plt.figure(figsize=figsize)
		ax_mean = plt.subplot()
	else:
		ax_mean = ax
	# Mean plot
	ax_mean.set_xlabel(x_label)
	ax_mean.plot(x, means, label="mean", color=color_mean, marker="o")
	ax_mean.set_ylabel("Mean", color=color_mean)
	ax_mean.tick_params(axis='y', labelcolor=color_mean)
	ax_mean.ticklabel_format(axis='y', style="sci", scilimits=(-3, 3))
	# Variance plot
	ax_variance = ax_mean.twinx()
	ax_variance.plot(x, variances, label="variance", color=color_variance)
	ax_variance.set_ylabel("Variance", color=color_variance)
	ax_variance.tick_params(axis='y', labelcolor=color_variance)
	ax_variance.ticklabel_format(axis='y', style="sci", scilimits=(-3, 3))
	# Plot handlers
	lines_mean, labels_mean = ax_mean.get_legend_handles_labels()
	lines_variance, labels_variance = ax_variance.get_legend_handles_labels()
	ax_mean.legend(lines_mean + lines_variance, labels_mean + labels_variance, loc="upper left")
	plt.tight_layout()
	plt.title(title)
	# Save and show figure
	if save_path is not None:
		plt.savefig(save_path)
	if is_show:
		plt.show()
	plt.close()
	return means, variances

# Visualize tensor value distribution by histogram
# @param tensor: torch.Tensor or numpy.ndarray
# @param ax: Matplotlib subplot object
# @param figsize: [Tuple[Int, Int]] Only take effect when `ax` is None
# @param bins: [Int] The number of bins of histogram
# @param save_path: [Str] Figure save path
# @param is_show: [Boolean] Whether to show figure
# @param x_label: [Str]
# @param y_label: [Str]
# @param title: [Str] Figure title
# @param **kwargs: [Dict] Other keyword arguments for `ax.hist`
def plot_tensor_histogram(tensor, *,
						  ax = None,
						  figsize=(10, 8),
						  bins = 50,
						  save_path = None,
						  is_show = True,
						  x_label = "Value",
						  y_label = "Frequency",
						  title = "Tensor Value Distribution",
						  **kwargs,
						  ):
	if hasattr(tensor, "numpy"):
		data = tensor.cpu().detach().numpy().flatten()
	else:
		data = numpy.array(tensor).flatten()
	if ax is None:
		plt.figure(figsize=figsize)
		ax = plt.subplot()
	mean = round(data.mean().item(), 4)
	variance = round(data.var().item(), 4)
	ax.hist(data, bins=bins, edgecolor="black", alpha=0.7, label=f"Mean: {mean}\nVar: {variance}")
	ax.set_title(title)
	ax.set_xlabel(x_label)
	ax.set_ylabel(y_label)
	ax.grid(axis='y', alpha=.5)
	# `scilimits = (m, n)` refers to use Scientific Notation for value
	ax.ticklabel_format(axis='y', style="sci", scilimits=(-2, 2))
	ax.legend(), ax.grid(True, alpha=.3)
	if save_path is not None:
		plt.savefig(save_path)
	if is_show:
		plt.show()
	plt.close()

# Visualize tensor value distribution by heatmap
# @param tensor: torch.Tensor or numpy.ndarray
# @param ax: Matplotlib subplot object
# @param figsize: [Tuple[Int, Int]] Only take effect when `ax` is None
# @param title: [Str]
# @param save_path: [Str] Figure save path
# @param is_show: [Boolean] Whether to show figure
# @param x_label: [Str]
# @param y_label: [Str]
# @param title: [Str] Figure title
# @param heatmap_kwargs: [Dict] Keyword arguments of `sns.heatmap`
# - cmap: [Str] e.g. "binary", "viridis", "coolwarm", "Greys", "gray", "gist_gray", "gist_yarg"
# - annot: [Boolean] Whether to show value in grid cell
# - fmt: [Str] Value formatter, e.g. ".2f"
# - cbar: [Boolean] whether to show color bar
def plot_tensor_heatmap(tensor, *,
						ax = None,
						figsize = (10, 8),
						save_path = None,
						is_show = True,
						x_label = "Column",
						y_label = "Row",
						title = "Tensor Value Heatmap",
						heatmap_kwargs = {
							"cmap": "binary",
							"annot": False,
							"fmt": ".2f",
							"cbar": True,
						},
						):
	if hasattr(tensor, "numpy"):
		data = tensor.cpu().detach().numpy()
	else:
		data = numpy.array(tensor)
	assert data.ndim == 2
	if ax is None:
		plt.figure(figsize=figsize)
		ax = plt.subplot()
	sns.heatmap(data, ax=ax, **heatmap_kwargs)
	ax.set_title(title)
	ax.set_xlabel(x_label)
	ax.set_ylabel(y_label)
	if save_path is not None:
		plt.savefig(save_path)
	if is_show:
		plt.show()
	plt.close()

# Plot dynamics of trainer_state of `transformers.Trainer`, e.g. train/eval loss, train/eval mean token accuracy, train/eval entropy
# @param trainer_state_paths: [List[Str]] File paths of several `trainer_state.json` to be compared
# @param trainer_state_names: [List[Str]] Default None refers to [0, 1, 2, ...]
# @param plot_index_names: [List[Str]] index to plot in `log_history`, e.g. `["loss", "mean_token_accuracy", "entropy"]`
# @param eval_keys: List[Str]
# - When keyword argument `eval_dataset` of `SFTTrainer` is only one dataset, then refers to [str()]
# - When keyword argument `eval_dataset` of `SFTTrainer` is a [Dict] of several datasets, then refers to the keys of `eval_dataset`, usually ["eval_0", "eval_1", ...]
# @param x_index_name: [Str] index to plot in `log_history`, "epoch" or "step"
# @param figure_size: [Int] Figure size of width or height (usually the same)
# @param save_path: [Str] Figure save path
# @param is_show: [Boolean] Whether to show figure
def plot_trainer_state(trainer_state_paths,
					   trainer_state_names = None,
					   plot_index_names = ["loss", "mean_token_accuracy", "entropy"],
					   eval_keys = [str()],
					   x_index_name = "epoch",
					   figure_size = 5,
					   save_path = None,
					   is_show = True,
					   ):
	if trainer_state_names is None:
		trainer_state_names = list(map(lambda _x: _x.replace('\\', '/').split('/')[-1], trainer_state_paths[:]))
	for i, (trainer_state_name, trainer_state_path) in enumerate(zip(trainer_state_names, trainer_state_paths)):
		with open(trainer_state_path, 'r', encoding="utf8") as f:
			data = json.load(f)
		log_history = data["log_history"]	# List[Dict]
		x_data_train = list()
		y_data_train = {index_name: list() for index_name in plot_index_names}
		x_data_eval_dict = dict()
		y_data_eval_dict = dict()
		for eval_key in eval_keys:
			x_data_eval_dict[eval_key] = list()
			y_data_eval_dict[eval_key] = {index_name: list() for index_name in plot_index_names}
		for entry in log_history:
			if "loss" in entry:
				# This is a train log entry
				x_data_train.append(entry[x_index_name])	# Epoch 0, 1, 2, ...
				for index_name in plot_index_names:
					y_data_train[index_name].append(entry[index_name])
			if f"eval_entropy" in entry:
				# This is an eval log entry
				eval_key_flag = False
				for eval_key in eval_keys:
					if eval_key:
						# `eval_key` is not an empty string
						if f"eval_{eval_key}_loss" in entry:
							eval_key_flag = True
							x_data_eval_dict[eval_key].append(entry[x_index_name])
							for index_name in plot_index_names:
								if index_name in ["entropy", "mean_token_accuracy", "num_tokens"]:
									y_data_eval_dict[eval_key][index_name].append(entry[f"eval_{index_name}"])
								else:
									y_data_eval_dict[eval_key][index_name].append(entry[f"eval_{eval_key}_{index_name}"])
							break
					else:
						# `eval_key` is an empty string, i.e. keyword argument `eval_dataset` of `SFTTrainer` is only one dataset
						if f"eval_loss" in entry:
							eval_key_flag = True
							x_data_eval_dict[eval_key].append(entry[x_index_name])
							for index_name in plot_index_names:
								y_data_eval_dict[str()][index_name].append(entry[f"eval_{index_name}"])
							break
				if not eval_key_flag:
					raise Exception(f"Cannot find any eval enties according to {eval_keys}")
		if i == 0:
			# Initialize canvas at the first trainer_states.json
			if y_data_eval_dict[eval_keys[0]][plot_index_names[0]]:
				nrows, ncols = len(eval_keys) + 1, len(plot_index_names)	# nrows > 1 ==> train & eval
			else:
				nrows, ncols = 1, len(plot_index_names)	# nrows = 1 ==> train only
			fig, axes = plt.subplots(
				nrows = nrows,
				ncols = ncols,
				figsize = (figure_size * 1.2 * ncols, figure_size * nrows),
			)
		# Train plot
		for j in range(ncols):
			y_index_name = plot_index_names[j]	# loss, mean_token_accuracy, entropy, etc
			if nrows == 1 and ncols == 1:
				target_ax = axes
			elif ncols == 1:
				target_ax = axes[0]
			elif nrows == 1:
				target_ax = axes[j]
			else:
				target_ax = axes[0][j]
			target_ax.plot(x_data_train, y_data_train[y_index_name], label=trainer_state_name)
			target_ax.set_xlabel(x_index_name), target_ax.set_ylabel(y_index_name), target_ax.set_title(f"{y_index_name} by {x_index_name}")
			target_ax.legend(), target_ax.grid(True, alpha=.3)
			if nrows > 1:
				for k, eval_key in enumerate(eval_keys):
					target_ax = axes[k + 1] if ncols == 1 else axes[k + 1][j]
					target_ax.plot(x_data_eval_dict[eval_key], y_data_eval_dict[eval_key][y_index_name], label=trainer_state_name)
					target_ax.set_xlabel(x_index_name), target_ax.set_ylabel(y_index_name), target_ax.set_title(f"{eval_key}_{y_index_name} by {x_index_name}")
					target_ax.legend(), target_ax.grid(True, alpha=.3)
	if save_path is not None:
		plt.savefig(save_path)
	if is_show:
		plt.show()
	plt.close()

# Plot statistics of SVD of LoRA adapter, e.g. LoRA rank, nuclear_norm
# @param adapter_path: [Str] The `output_dir` of `transformers.trainer`
# @param thresholds: [List[Float]] Different thresholds used to control approximate rank
# @param plot_index_names: [List[Str]] index to plot, e.g. `["nuclear_norm", "rank"]`
# @param device: [Str] "cpu" or "cuda"
# @param figure_size: [Int] Figure size of width or height (usually the same)
# @param save_path: [Str] Figure save path
# @param is_show: [Boolean] Whether to show figure
# @return summary: [Dict] key is Str[module_name_prefix], value is the Tuple containing the several statistics of the given adapter
def plot_lora_adapter_statistics(adapter_path,
								 thresholds = [.1, .05, .01],
								 target_modules = ["q_proj", "k_proj", "v_proj"],
								 plot_index_names = ["nuclear_norm", "rank"],
								 device = "cpu",
								 figure_size = 5,
								 save_path = None,
								 is_show = True,
								 ):
	with safe_open(adapter_path, framework="pt", device=device) as f:
		summary = dict()
		keys = sorted(f.keys(), key = lambda _x: (int(_x.split('.')[4]), _x.split('.')[6], _x.split('.')[7]), reverse=False)
		for key_A, key_B in zip(keys[:-1][::2], keys[1:][::2]):
			module_A_prefix = '.'.join(key_A.split('.')[:-2])
			module_B_prefix = '.'.join(key_B.split('.')[:-2])
			assert module_A_prefix == module_B_prefix, f"{key_A} v.s. {key_B}"
			lora_A, lora_B = f.get_tensor(key_A), f.get_tensor(key_B)
			assert lora_A.size(0) == lora_B.size(1), f"{lora_A.size()} v.s. {lora_B.size()}"
			BA = lora_B @ lora_A
			U, S, V = torch.svd(BA)
			nuclear_norm = torch.sum(S)
			ranks = []
			for threshold in thresholds:
				total_singular_values = 0
				for i, s in enumerate(S):
					total_singular_values += s
					if total_singular_values / nuclear_norm >= 1 - threshold:
						ranks.append(i + 1)
						break
			assert not module_A_prefix in summary
			summary[module_A_prefix] = (nuclear_norm, ranks)
	plot_data = {
		target_module: {
			"nuclear_norm": list(),
			"rank": {threshold: list() for threshold in thresholds},
		}
		for target_module in target_modules
	}
	for module_name, (nuclear_norm, ranks) in summary.items():
		for target_module in target_modules:
			if target_module in module_name:
				plot_data[target_module]["nuclear_norm"].append(nuclear_norm)
				for i, threshold in enumerate(thresholds):
					plot_data[target_module]["rank"][threshold].append(ranks[i])

	nrows, ncols = len(target_modules), len(plot_index_names)
	fig, axes = plt.subplots(
		nrows = nrows,
		ncols = ncols,
		figsize = (figure_size * 1.2 * ncols, figure_size * nrows),
	)
	for i, target_module in enumerate(target_modules):
		for j, plot_index_name in enumerate(plot_index_names):
			if nrows == 1 and ncols == 1:
				target_ax = axes
			elif nrows == 1:
				target_ax = axes[j]
			elif ncols == 1:
				target_ax = axes[i]
			else:
				target_ax = axes[i, j]
			if plot_index_name in ["rank"]:
				# rank has several series of data
				for threshold in thresholds:
					target_ax.plot(plot_data[target_module][plot_index_name][threshold], label=f"threshold: {threshold}", marker='o')
				target_ax.set_xlabel("Layer #"), target_ax.set_ylabel("Value"), target_ax.set_title(f"{plot_index_name} of {target_module}")
			elif plot_index_name in ["nuclear_norm"]:
				target_ax.plot(plot_data[target_module][plot_index_name], label=f"threshold: {threshold}", marker='o')
				target_ax.set_xlabel("Layer #"), target_ax.set_ylabel("Value"), target_ax.set_title(f"{plot_index_name} of {target_module}")
			else:
				raise Exception(f"Unknown index: {plot_index_name}")
			target_ax.legend(), target_ax.grid(True, alpha=.3)
	if save_path is not None:
		plt.savefig(save_path)
	if is_show:
		plt.show()
	plt.close()
	return summary


# Bar plot the results of evaluation, i.e. the dumped JSON of evaluator pipelines
# @param result_paths: [List[Str]] JSON paths of evaluation results
# @param result_names: [List[Str]|NoneType] Give name to each result file, default use the JSON filename
# @param metric_names: [List[Str]] Metrics to plot, e.g. ["token_accuracy", "perplexity", "bleu_3", "rouge_3", "rouge_l", "rouge_w"]
# @param plot_index_names: [List[Str]] Index to plot, e.g. ["mean", "std"]
# @param figure_size: [Int] Figure size of width or height (usually the same)
# @param width: [Float] Width of each bar
# @param save_path: [Str] Figure save path
# @param is_show: [Boolean] Whether to show figure
def plot_evaluation_results(result_paths,
							result_names = None,
							metric_names = ["token_accuracy", "perplexity", "bleu_3", "rouge_3", "rouge_l", "rouge_w"],
							plot_index_names = ["mean", "std"],
							figure_size = 5,
							width = .25,
							save_path = None,
							is_show = True,
							):
	summary = dict()
	metric_names_map = dict()
	if result_names is None:
		result_names = list(map(lambda _x: _x.replace('\\', '/').split('/')[-1], result_paths[:]))
	for result_path in result_paths:
		with open(result_path, 'r', encoding="utf8") as f:
			data = json.load(f)
		for metric_name in metric_names:
			metric_data = data[metric_name]
			if metric_name in ["token_accuracy", "perplexity", "bleu_3"]:
				# Single value
				if metric_name not in summary:
					summary[metric_name] = {"mean": list(), "std": list()}
				summary[metric_name]["mean"].append(metric_data["population_mean"])
				summary[metric_name]["std"].append(metric_data["population_std"])
				metric_names_map[metric_name] = [metric_name]
			elif metric_name in ["rouge_3", "rouge_l", "rouge_w"]:
				# Multiple values: Precision Recall F1-score
				if f"{metric_name}_p" not in summary:
					summary[f"{metric_name}_p"] = {"mean": list(), "std": list()}
				if f"{metric_name}_r" not in summary:
					summary[f"{metric_name}_r"] = {"mean": list(), "std": list()}
				if f"{metric_name}_f1" not in summary:
					summary[f"{metric_name}_f1"] = {"mean": list(), "std": list()}
				summary[f"{metric_name}_p"]["mean"].append(metric_data["population_mean"][0])
				summary[f"{metric_name}_p"]["std"].append(metric_data["population_std"][0])
				summary[f"{metric_name}_r"]["mean"].append(metric_data["population_mean"][1])
				summary[f"{metric_name}_r"]["std"].append(metric_data["population_std"][1])
				summary[f"{metric_name}_f1"]["mean"].append(metric_data["population_mean"][2])
				summary[f"{metric_name}_f1"]["std"].append(metric_data["population_std"][2])
				metric_names_map[metric_name] = [f"{metric_name}_p", f"{metric_name}_r", f"{metric_name}_f1"]
			else:
				raise NotImplementedError(metric_name)
	nrows, ncols = len(metric_names_map), len(plot_index_names)
	fig, axes = plt.subplots(
		nrows = nrows,
		ncols = ncols,
		figsize = (figure_size * 1.2 * ncols, figure_size * nrows),
	)
	n_series = len(result_names)
	for i, (metric_name, mapped_metric_names) in enumerate(metric_names_map.items()):
		for j, plot_index_name in enumerate(plot_index_names):
			if nrows == 1 and ncols == 1:
				target_ax = axes
			elif nrows == 1:
				target_ax = axes[j]
			elif ncols == 1:
				target_ax = axes[i]
			else:
				target_ax = axes[i, j]
			n_groups = len(mapped_metric_names)
			x = numpy.arange(n_groups)
			for k, result_name in enumerate(result_names):
				data = [summary[mapped_metric_name][plot_index_name][k] for mapped_metric_name in mapped_metric_names]
				offset = width * (k - (n_series - 1) / 2)
				target_ax.bar(x + offset, data, width, label=result_name, alpha=.5)
			target_ax.set_xticks(x, mapped_metric_names), target_ax.set_ylabel("Value"), target_ax.set_title(f"{plot_index_name} of {metric_name}")
			target_ax.legend(), target_ax.grid(True, alpha=.3)
	if save_path is not None:
		plt.savefig(save_path)
	if is_show:
		plt.show()
	plt.close()
