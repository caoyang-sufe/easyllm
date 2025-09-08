# -*- coding: utf8 -*-
# @author: caoyang
# @email: caoyang@stu.sufe.edu.cn

import torch

from src.tools.transformers import greedy_decode


# @param hook_data: hook_data
# @param num_hidden_layers: [Int] The number of hidden layers, usually comes from `model.config.num_hidden_layers`
def visualize_layer_inputs_and_outputs(hook_data,
									   hook_data_path = None,
									   max_length,
									   num_hidden_layers,
									   module_name_formatter = "model.layers[{}]",
									   ):
	for i in range(max_length):
		pass
		
	for j in range()

# Analyze the reasoning dynamics in single reasoning process
# @param model: 
def layer_dynamics_in_reasoning(model,
								tokenizer,
								prompt,
								max_length,
								layer_name_formatter,
								layer_name_formatter,
								
								num_hidden_layers = None,
								):  
	if num_hidden_layers is None:
		num_hidden_layers = model.config.num_hidden_layers
	for i in range(i):


# Compare hook_data by module names
# @param hook_datas: List[<hook_data object>] of length 2, currently
def compare_layer_dynamics(hook_datas = None,
						   hook_data_paths = None,
						   forward_hook_module_names,
						   figure_names = None,
						   ):
	regex = re.compile("\[\d+\]", re.I)	# Used to match
	assert hook_datas is not None or hook_data_paths is not None
	if hook_datas is None:
		hook_datas = [torch.load(hook_data_path) for hook_data_path in hook_data_paths]
	
	for i, hook_data_group in enumerate(zip(*hook_datas)):
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
				# Process Input
				input_data_group = [data[module_name].get("input", data[module_name]["args"]) for data in hook_data_group]
				for j, input_data in enumerate(input_data_group):     
					# Assertation for ensuring data format of inputs
					assert len(input_data) == 1 and isinstance(input_data[0], tuple)
					if len(input_data[0]) > 1:
						logging.warning(f"Input data {j} has more than 1 components: {len(input_data[0])}")
				input_tensors = [input_data[0][0].float() for input_data in input_data_group]
				for j, input_tensor in enumerate(input_tensors):
					logging.info(f"Size of input tensor {j}: {input_tensor.size()}")
				# Process Output
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
				
				input_diff = torch.norm(input_tensors[0] - input_tensors[1], p="fro")
				output_diff = torch.norm(output_tensors[0] - output_tensors[1], p="fro")
				avg_input_diff = input_diff / input_tensors[0].numel()
				avg_output_diff = output_diff / output_tensors[0].numel()
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
	
	
def skip_layer_generation():
	
	pass
	

