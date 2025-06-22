# -*- coding: utf8 -*-
# @author: caoyang
# @email: caoyang@stu.sufe.edu.cn

import json
import torch
import numpy
import seaborn as sns
from matplotlib import pyplot as plt

# @param tensor: torch.Tensor or numpy.ndarray
# @param figsize: [Tuple[Int, Int]]
# @param bins: [Int] the number of bins of histogram
# @param title: [Str]
# @param xlabel: [Str]
# @param ylabel: [Str]
def plot_tensor_histogram(tensor,
						  ax = None,
						  figsize=(10, 8),
						  bins = 50,
						  title = "Tensor Value Distribution", 
						  xlabel = "Value", 
						  ylabel = "Frequency",
						  save_path = None,
						  is_show = True,
						  ):
	if hasattr(tensor, "numpy"):
		data = tensor.cpu().detach().numpy().flatten()
	else:
		data = numpy.array(tensor).flatten()
	plt.figure(figsize=figsize)
	if ax is None:
		plt.hist(data, bins=bins, edgecolor="black", alpha=0.7)
		plt.title(title)
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.grid(axis='y', alpha=.75)
	else:
		ax.hist(data, bins=bins, edgecolor="black", alpha=0.7)
		ax.set_title(title)
		ax.set_xlabel(xlabel)
		ax.set_ylabel(ylabel)
		ax.grid(axis='y', alpha=.75)		
	if save_path is not None:
		plt.savefig(save_path)
	if is_show:
		plt.show()
		plt.close()

# @param tensor: torch.Tensor or numpy.ndarray
# @param figsize: [Tuple[Int, Int]]
# @param title: [Str]
# @param cmap: [Str] e.g. "viridis", "coolwarm"
# @param annot: [Boolean] whether to show value in grid cell
# @param fmt: [Str] value formatter, e.g. ".2f"
# @param cbar: [Boolean] whether to show color bar
def plot_tensor_heatmap(tensor,
						ax = None,
						figsize = (10, 8),
						title = "Tensor Heatmap",
						cmap = "viridis", 
						annot = False,
						fmt = ".2f",
						cbar = True,
						save_path = 
						):
	if hasattr(tensor, "numpy"):
		data = tensor.cpu().detach().numpy()
	else:
		data = numpy.array(tensor)
	assert data.ndim == 2
	plt.figure(figsize=figsize)
	if ax is None:
		ax = sns.heatmap(data, cmap=cmap, annot=annot, fmt=fmt, cbar=cbar)
	else:
		sns.heatmap(data, cmap=cmap, annot=annot, fmt=fmt, cbar=cbar, ax=ax)
	ax.set_title(title)
	plt.xlabel("Columns")
	plt.ylabel("Rows")
	if save_path is not None:
		plt.savefig(save_path)		
	if is_show:
		plt.show()
		plt.close()

# Plot dynamics of TRL trainer state
def plot_trl_dynamics(trainer_state_path):
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
	plt.figure(figsize=(8, 8))
	ax_1 = plt.subplot(2, 2, 1)
	ax_2 = plt.subplot(4, 2, 2)
	ax_3 = plt.subplot(4, 2, 4)
	ax_4 = plt.subplot(2, 2, 3)
	ax_5 = plt.subplot(2, 2, 4)

	ax_1.plot(steps, policy_loss, label="Policy Loss")
	ax_1.plot(steps, value_loss, label="Value Loss", linestyle="--")
	ax_1.set_xlabel("Step"), ax_1.set_ylabel("Loss"), ax_1.legend()
	ax_1.set_title("Policy and Value Loss")
	# ------------------------------------------------------------------
	ax_2.plot(steps, kls, label="objective/kl")
	ax_2.set_xlabel("Step"), ax_2.set_ylabel("KL"), ax_2.legend()
	ax_2.set_title("KL Curve")
	# ------------------------------------------------------------------
	ax_3.plot(steps, entropys, label="objective/entropy")
	ax_3.set_xlabel("Step"), ax_3.set_ylabel("Entropy"), ax_3.legend()
	ax_3.set_title("Entropy Curve")
	# ------------------------------------------------------------------
	ax_4.plot(steps, lrs, label="Learning Rate")
	ax_4.set_xlabel("Step"), ax_4.set_ylabel("Learning Rate"), ax_4.legend()
	ax_4.set_title("Learning Rate Curve")
	# ------------------------------------------------------------------
	ax_5.plot(steps, non_score_rewards, label="objective/non_score_reward", linestyle="--")
	ax_5.plot(steps, rlhf_rewards, label="objective/rlhf_reward", linestyle="--")
	ax_5.plot(steps, scores, label="objective/scores")
	ax_5.set_xlabel("Step"), ax_5.set_ylabel("Score/Reward"), ax_5.legend()
	ax_5.set_title("Reward and Score")
	plt.show()

# Plot Mean and Variance of Tensor
def plot_tensor_mean_and_variance(tensor):
	x = np.linspace(0, 10, 100)
	y1 = np.sin(x)
	y2 = np.exp(x) * 0.1
	fig, ax1 = plt.subplots(figsize=(8, 4))
	ax1.plot(x, y1, color="blue", label="sin(x)")
	ax1.set_xlabel("X Axis")
	ax1.set_ylabel("Y1 (sin(x))", color="blue")
	ax1.tick_params(axis='y', labelcolor="blue")
	ax2 = ax1.twinx()
	ax2.plot(x, y2, color="red", label="0.1 * exp(x)")
	ax2.set_ylabel("Y2 (0.1 * exp(x))", color="red")
	ax2.tick_params(axis='y', labelcolor="red")
	lines1, labels1 = ax1.get_legend_handles_labels()
	lines2, labels2 = ax2.get_legend_handles_labels()
	ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
	plt.tight_layout()
	plt.title("Dual Y-Axis Example")
	plt.show()

	
if __name__ == "__main__":
	fp = r"C:\Users\caoyang\AppData\Local\Temp\fz3temp-2\trainer_state.json"
	fp = r"D:\code\trainer_state.json"
	plot_trl_dynamics(fp)
	
