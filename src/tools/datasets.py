# -*- coding: utf-8 -*-
# @author: caoyang
# @email: caoyang@stu.sufe.edu.cn
# Torch operations

from datasets import load_dataset, DatasetDict

# Add several splits to the given dataset
# @param dataset: [DatasetDict] Given dataset
# @param dataset_splits: [List[Dataset]] Several splits to be added to raw dataset
# @param split_names: [List[Str]] Names of each split in raw dataset
# @return: New DatasetDict
def add_dataset_split(dataset, dataset_splits, split_names=None):
	if split_names is None:
		split_names = [f"split_{i}" for i in range(len(dataset_splits))]
	new_dataset_dict = {key: dataset[key] for key in dataset.keys()}
	for split_name, dataset_split in zip(split_names, dataset_splits):
		new_dataset_dict[split_name] = dataset_split
	return DatasetDict(new_dataset_dict)

