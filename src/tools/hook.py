# -*- coding: utf8 -*-
# @author: caoyang
# @email: caoyang@stu.sufe.edu.cn

import torch
from functools import wraps
from torch.nn import functional as F

# Forward hook decorator
# @params module_names: List[Str], e.g. ["model.layers[0].self_attn.q_proj", "model.layers[0].self_attn.k_proj"]
def register_forward_hook_decorator(module_names):
	# @param func: the call of `func` must include a keyword argument "model"
	def decorator(func):
		@wraps(func)
		def wrapper(*args, **kwargs):
			model = kwargs["model"]
			hook_data = dict()
			hook_handles = list()
			for module_name in module_names:
				def _make_hook(_module_name):
					hook_data[module_name] = dict()
					# @param _module: `f"Module: {_module.__class__.__name__}"`
					# @param _inputs: Tuple[torch.FloatTensor], `f"Input shapes: {[x.shape for x in _inputs]}"`
					# @param _outputs: torch.FloatTensor/Tuple[torch.FloatTensor], `f"Output shape: {_outputs.shape}"`
					def _hook(_module, _input, _output):
						hook_data[_module_name]["output"] = _output
						hook_data[_module_name]["input"] = _input
					return _hook
				hook_handles.append(eval(f"model.{module_name}").register_forward_hook(_make_hook(module_name)))
			try:
				func_return = func(*args, **kwargs)
				func_return.hook_outputs = hook_data	# Attach hook data to function returns
				return func_return
			finally:
				for hook_handle in hook_handles:
					hook_handle.remove()
		return wrapper
	return decorator

# Backward hook decorator (for gradient)
# @params module_names: List[Str], e.g. ["model.layers[0].self_attn.q_proj", "model.layers[0].self_attn.k_proj"]
def register_backward_hook_decorator(module_names):
	# @param func: the call of `func` must include a keyword argument "model"
	def decorator(func):
		@wraps(func)
		def wrapper(*args, **kwargs):
			model = kwargs["model"]
			hook_data = dict()
			hook_handles = list()
			for module_name in module_names:
				def _make_hook(_module_name):
					hook_data[module_name] = {"input": list(), "output": list()}
					# @param _module: `f"Module: {_module.__class__.__name__}"`
					# @param _inputs: Tuple[torch.FloatTensor], `f"Input shapes: {[x.shape for x in _inputs]}"`
					# @param _outputs: Tuple`f"Output shape: {_outputs.shape}"`
					def _hook(_module, _input, _output):
						hook_data[_module_name]["input"].append(_input.detach().clone())
						hook_data[_module_name]["output"].append(_output.detach().clone())
					return _hook
				hook_handles.append(eval(f"model.{module_name}").register_forward_hook(_make_hook(module_name)))
			try:
				func_return = func(*args, **kwargs)
				func_return.hook_outputs = hook_data	# Attach hook data to function returns
				return func_return
			finally:
				for hook_handle in hook_handles:
					hook_handle.remove()
		return wrapper
	return decorator
