# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@stu.sufe.edu.cn

import torch
from functools import wraps
from torch.nn import functional as F

# Forward hook decorator: Register this decorator to any functions with keyword argument `model`
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
					hook_data[module_name] = {"args": list(), "kwargs": list(), "output": list()}
					# @param _module: `f"Module: {_module.__class__.__name__}"`
					# @param _args: [Tuple] positional arguments of module inputs
					# @param _kwargs: [Dict] keyword arguments of module inputs
					# @param _outputs: the return of module.forward, usually in format of torch.FloatTensor or Tuple[torch.FloatTensor]
					def _hook(_module, _args, _kwargs, _outputs):
						hook_data[_module_name]["args"].append(_args)
						hook_data[_module_name]["kwargs"].append(_kwargs)
						hook_data[_module_name]["output"].append(_outputs)
					return _hook
				hook_handles.append(eval(f"model.{module_name}").register_forward_hook(_make_hook(module_name), with_kwargs=True))
			try:
				func_return = func(*args, **kwargs)
				func_return.hook_outputs = hook_data	# Attach hook data to function returns
				return func_return
			finally:
				for hook_handle in hook_handles:
					hook_handle.remove()
		return wrapper
	return decorator

# Backward hook decorator (for gradient): Register this decorator to any functions with keyword argument `model`
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
					# @param _outputs: Tuple[torch.FloatTensor]: `f"Output shapes: {[x.shape for x in _outputs]}"`
					def _hook(_module, _inputs, _outputs):
						hook_data[_module_name]["input"].append(_inputs)
						hook_data[_module_name]["output"].append(_outputs)
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
