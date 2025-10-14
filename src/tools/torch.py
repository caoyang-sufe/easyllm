# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@stu.sufe.edu.cn
# Torch operations

import torch
from torch.nn import functional as F

# Calculate cosine similarity by filtering outlier
# @param x: [torch.Tensor]
# @param y: [torch.Tensor]
# @param filter_outlier: [Float] range from [0, 1)
def robust_cosine_similarity(x, y, outlier_ratio = .1):
	x, y = x.flatten(), y.flatten()
	assert x.size(0) == y.size(0)
	abs_diff = torch.abs(x - y)
	k = int(len(abs_diff) * (1 - outlier_ratio))
	_, indices = torch.topk(abs_diff, k=k, largest=False)
	x_filtered, y_filtered = x[indices], y[indices]
	similarity = F.cosine_similarity(x_filtered, y_filtered, dim=0).item()
	return similarity
	
# Calculate correlation coefficient by filtering outlier
# @param x: [torch.Tensor]
# @param y: [torch.Tensor]
# @param outlier_ratio: [Float] range from [0, 1)
def robust_corrcoef(x, y, outlier_ratio = .1):
	x, y = x.flatten(), y.flatten()
	assert x.size(0) == y.size(0)
	abs_diff = torch.abs(x - y)
	k = int(len(abs_diff) * (1 - outlier_ratio))
	_, indices = torch.topk(abs_diff, k=k, largest=False)
	x_filtered, y_filtered = x[indices], y[indices]
	corrcoef = torch.corrcoef(torch.stack([x_filtered, y_filtered]))[0, 1].item()
	return corrcoef
