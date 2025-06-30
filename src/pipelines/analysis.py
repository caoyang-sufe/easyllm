# -*- coding: utf8 -*-
# @author: caoyang
# @email: caoyang@stu.sufe.edu.cn

import torch

from src.tools.transformers import greedy_decode


def greedy_decode(model,
				  tokenizer,
				  prompt, 
				  max_length,
				  device = "cuda",
				  use_kv_cache = True,
				  forward_hook_module_names = None,
				  backward_hook_module_names = None,
				  ):


# 
def layer_dynamics_in_reasoning(model,
								tokenizer,
								prompt,
								max_length,
								layer_name_formatter,
								):
	for i in range


# Compare  hook_data by module names
# @param hook_data_path
def compare_layer_dynamics(hook_datas,
							  hook_data_paths = None,
							  
							  hook_data_path_2,
							  forward_hook_module_names
							  forward_hook_module_names,
							  figure_names = None,
							  pivot_at = 0,
							  ):
	regex = re.compile("\[\d+\]", re.I)
	
	
	
	hook_data_1 = torch.load(hook_data_path_1)
	hook_data_2 = torch.load(hook_data_path_2)
	for i, (data_1, data_2) in enumerate(zip(hook_data_1, hook_data_2)):
		if i != pivot_at:
			# Care about the pivot token only
			continue
		if figure_names is None:
			diff_dict = {
				"embed_tokens": {"input": [], "output": []},
				"norm": {"input": [], "output": []},
				"q_proj": {"input": [], "output": []},
				"k_proj": {"input": [], "output": []},
				"v_proj": {"input": [], "output": []},
				"layers": {"input": [], "output": []},
			}
		else:
			diff_dict = {figure_name: {"input": [], "output": []} for figure_name in figure_names}
		for module_name in forward_hook_module_names:
			module_name_suffix = module_name.split('.')[-1]
			module_name_suffix = regex.sub(str(), module_name_suffix)
			if module_name_suffix in diff_dict:
				input_data_1 = data_1[module_name].get("input", data_1[module_name]["args"])
				input_data_2 = data_1[module_name].get("input", data_2[module_name]["args"])
				output_data_1 = data_1[module_name]["output"]
				output_data_2 = data_2[module_name]["output"]      
				# Assertation for ensuring data format
				assert len(input_data_1) == 1 and isinstance(input_data_1[0], tuple)
				assert len(input_data_2) == 1 and isinstance(input_data_2[0], tuple)
				if len(input_data_1[0]) > 1:
					print(f"*** Warning for input data 1: {len(input_data_1[0])} ***")
				if len(input_data_2[0]) > 1:
					print(f"*** Warning for input data 1: {len(input_data_2[0])} ***")
				input_tensor_1 = input_data_1[0][0].float()
				input_tensor_2 = input_data_2[0][0].float()
				# print(input_tensor_1.size(), input_tensor_2.size())
				# ---
				assert len(output_data_1) == 1
				if isinstance(output_data_1[0], torch.Tensor):
					output_tensor_1 = output_data_1[0]
				else:
					assert isinstance(output_data_1[0], tuple)
					if len(output_data_1[0]) > 1:
						print(f"*** Warning for output data 1: {len(output_data_1[0])} ***")
					output_tensor_1 = output_data_1[0][0]
				if isinstance(output_data_2[0], torch.Tensor):
					output_tensor_2 = output_data_2[0]
				else:
					assert isinstance(output_data_2[0], tuple)
					if len(output_data_2[0]) > 1:
						print(f"*** Warning for output data 2: {len(output_data_2[0])} ***")
					output_tensor_2 = output_data_2[0][0]
				input_diff = torch.norm(input_tensor_1 - input_tensor_2, p="fro")
				output_diff = torch.norm(output_tensor_1 - output_tensor_2, p="fro")
				avg_input_diff = input_diff / input_tensor_1.numel()
				avg_output_diff = output_diff / output_tensor_1.numel()
				# log_input_diff = torch.log(avg_input_diff)
				# log_output_diff = torch.log(avg_output_diff)
				diff_dict[module_name_suffix]["input"].append(avg_input_diff.item())
				diff_dict[module_name_suffix]["output"].append(avg_output_diff.item())
		
		figure_width = 5
		fig, axs = plt.subplots(1, len(diff_dict), figsize=((figure_width + 1) * len(diff_dict), figure_width))
		for i, key in enumerate(diff_dict):
			y_input = diff_dict[key]["input"]
			y_output = diff_dict[key]["output"]
			assert len(y_input) == len(y_output)
			x = range(len(y_input))
			if len(x) == 0:
				continue
			if len(diff_dict) == 1:
				axs.bar(x, y_input, label="input_diff", alpha=.5)
				axs.bar(x, y_output, label="output_diff", alpha=.5)
				axs.set_xlabel("Layer #"), axs.set_ylabel("Diff"), axs.set_title(f"Diff for {key}")
				axs.legend()
			else:
				axs[i].bar(x, y_input, label="input_diff", alpha=.5)
				axs[i].bar(x, y_output, label="output_diff", alpha=.5)
				axs[i].set_xlabel("Layer #"), axs[i].set_ylabel("Diff"), axs[i].set_title(f"Diff for {key}")
				axs[i].legend()
		plt.show(), plt.close()
	return diff_dict
