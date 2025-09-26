# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@stu.sufe.edu.cn

import json
import torch
import numpy
import seaborn as sns
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
	ax.legend()
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
		
# Plot dynamics of trainer_state of `transformers.Trainer`
# @param trainer_state_path: [Str] File path of trainer_state.json
# @param plot_index_names: [List[Str]] index to plot in `log_history`, e.g. `["loss", "mean_token_accuracy", "entropy"]`
# @param x_index_name: [Str] index to plot in `log_history`, "epoch" or "step"
# @param figsize: [Tuple[Int, Int]]
# @param save_path: [Str] Figure save path
# @param is_show: [Boolean] Whether to show figure
def plot_trainer_state(trainer_state_path,
					   plot_index_names = ["loss", "mean_token_accuracy", "entropy"],
					   x_index_name = "epoch",
					   figure_size = 5,
					   save_path = None,
					   is_show = True,
					   ):
	with open(trainer_state_path, 'r', encoding="utf8") as f:
		data = json.load(f)
	log_history = data["log_history"]	# List[Dict]
	x_data_train, x_data_eval = list(), list()
	y_data_train = {index_name: list() for index_name in plot_index_names}
	y_data_eval = {f"eval_{index_name}": list() for index_name in plot_index_names}
	for entry in log_history:
		if "loss" in entry:
			x_data_train.append(entry[x_index_name])
			for index_name in plot_index_names:
				y_data_train[index_name].append(entry[index_name])
		elif "eval_loss" in entry:
			x_data_eval.append(entry[x_index_name])
			for index_name in plot_index_names:
				y_data_eval[f"eval_{index_name}"].append(entry[f"eval_{index_name}"])

	if y_data_eval[f"eval_{plot_index_names[0]}"]:
		nrows, ncols = 2, len(plot_index_names)	# nrows = 2 ==> train & eval
	else:
		nrows, ncols = 1, len(plot_index_names)	# nrows = 2 ==> train & eval
	fig, axes = plt.subplots(
		nrows = nrows, 
		ncols = ncols,
		figsize = (figure_size * 1.2 * ncols, figure_size * nrows),
	)
	# Train plot
	for i in range(ncols):
		y_index_name = plot_index_names[i]
		if nrows == 1 and ncols == 1:
			target_ax = axes
		elif ncols == 1:
			target_ax = axes[0]
		elif nrows == 1:
			target_ax = axes[i]
		else:
			target_ax = axes[0][i]
		target_ax.plot(x_data_train, y_data_train[y_index_name], label=y_index_name)
		target_ax.set_xlabel(x_index_name), target_ax.set_ylabel(y_index_name), target_ax.set_title(f"{y_index_name} by {x_index_name}")
		target_ax.legend()
	if nrows == 2:
		# Eval plot
		for i in range(ncols):
			y_index_name = f"eval_{plot_index_names[i]}"
			if ncols == 1:
				target_ax = axes[1]
			else:
				target_ax = axes[1][i]
			target_ax.plot(x_data_eval, y_data_eval[y_index_name], label=y_index_name)
			target_ax.set_xlabel(x_index_name), target_ax.set_ylabel(y_index_name), target_ax.set_title(f"{y_index_name} by {x_index_name}")
			target_ax.legend()
	if save_path is not None:
		plt.savefig(save_path)		
	if is_show:
		plt.show()
		plt.close()

# Plot dynamics of trainer state of `trl.PPOTrainer`
# @param trainer_state_path: [Str] File path of trainer_state.json
# @param figsize: [Tuple[Int, Int]]
# @param save_path: [Str] Figure save path
# @param is_show: [Boolean] Whether to show figure
def plot_ppo_dynamics(trainer_state_path,
					  figsize = (8, 8),
					  save_path = None,
					  is_show = True,
					  ):
	with open(trainer_state_path, 'r', encoding="utf8") as f:
		data = json.load(f)
	log_history = data["log_history"]
	steps = [entry["step"] for entry in log_history]
	episodes = [entry["episode"] for entry in log_history]
	epochs = [entry["epoch"] for entry in log_history]
	policy_loss = [entry["loss/policy_avg"] for entry in log_history]
	value_loss = [entry["loss/value_avg"] for entry in log_history]
	lrs = [entry["lr"] for entry in log_history]
	entropys = [entry["objective/entropy"] for entry in log_history]
	kls = [entry["objective/kl"] for entry in log_history]
	non_score_rewards = [entry["objective/non_score_reward"] for entry in log_history]
	rlhf_rewards = [entry["objective/rlhf_reward"] for entry in log_history]
	scores = [entry["objective/scores"] for entry in log_history]
	
	plt.figure(figsize=figsize)
	ax_1 = plt.subplot(2, 2, 1)
	ax_2 = plt.subplot(4, 2, 2)
	ax_3 = plt.subplot(4, 2, 4)
	ax_4 = plt.subplot(2, 2, 3)
	ax_5 = plt.subplot(2, 2, 4)
	# ----
	ax_1.plot(steps, policy_loss, label="Policy Loss")
	ax_1.plot(steps, value_loss, label="Value Loss", linestyle="--")
	ax_1.set_xlabel("Step"), ax_1.set_ylabel("Loss"), ax_1.legend()
	ax_1.set_title("Policy and Value Loss")
	# ----
	ax_2.plot(steps, kls, label="objective/kl")
	ax_2.set_xlabel("Step"), ax_2.set_ylabel("KL"), ax_2.legend()
	ax_2.set_title("KL Curve")
	# ----
	ax_3.plot(steps, entropys, label="objective/entropy")
	ax_3.set_xlabel("Step"), ax_3.set_ylabel("Entropy"), ax_3.legend()
	ax_3.set_title("Entropy Curve")
	# ----
	ax_4.plot(steps, lrs, label="Learning Rate")
	ax_4.set_xlabel("Step"), ax_4.set_ylabel("Learning Rate"), ax_4.legend()
	ax_4.set_title("Learning Rate Curve")
	# ----
	ax_5.plot(steps, non_score_rewards, label="objective/non_score_reward", linestyle="--")
	ax_5.plot(steps, rlhf_rewards, label="objective/rlhf_reward", linestyle="--")
	ax_5.plot(steps, scores, label="objective/scores")
	ax_5.set_xlabel("Step"), ax_5.set_ylabel("Score/Reward"), ax_5.legend()
	ax_5.set_title("Reward and Score")
	# ----
	if save_path is not None:
		plt.savefig(save_path)		
	if is_show:
		plt.show()
		plt.close()

	
if __name__ == "__main__":
	pass
	
