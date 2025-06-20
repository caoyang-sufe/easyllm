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
						  figsize=(10, 8),
						  bins = 50,
						  title = "Tensor Value Distribution", 
						  xlabel = "Value", 
						  ylabel = "Frequency",
						  ):
	if hasattr(tensor, "numpy"):
		data = tensor.cpu().detach().numpy().flatten()
	else:
		data = numpy.array(tensor).flatten()
	plt.figure(figsize=figsize)
	plt.hist(data, bins=bins, edgecolor="black", alpha=0.7)
	plt.title(title)
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.grid(axis='y', alpha=.75)
	plt.show()

# @param tensor: torch.Tensor or numpy.ndarray
# @param figsize: [Tuple[Int, Int]]
# @param title: [Str]
# @param cmap: [Str] e.g. "viridis", "coolwarm"
# @param annot: [Boolean] whether to show value in grid cell
# @param fmt: [Str] value formatter, e.g. ".2f"
# @param cbar: [Boolean] whether to show color bar
def plot_tensor_heatmap(tensor,
						figsize = (10, 8),
						title = "Tensor Heatmap",
						cmap = "viridis", 
						annot = False,
						fmt = ".2f",
						cbar = True,
						):
	if hasattr(tensor, "numpy"):
		data = tensor.cpu().detach().numpy()
	assert data.dim() == 2
	plt.figure(figsize=figsize)
	ax = sns.heatmap(data, cmap=cmap, annot=annot, fmt=fmt, cbar=cbar)
	ax.set_title(title)
	plt.xlabel("Columns")
	plt.ylabel("Rows")
	plt.show()

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
	
if __name__ == "__main__":
	fp = r"C:\Users\caoyang\AppData\Local\Temp\fz3temp-2\trainer_state.json"
	fp = r"D:\code\trainer_state.json"
	plot_trl_dynamics(fp)
	