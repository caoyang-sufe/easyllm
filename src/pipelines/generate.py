# -*- coding: utf8 -*-
# @author: caoyang
# @email: caoyang@stu.sufe.edu.cn

import torch
import pandas
from src.tools.transformers import generate_token_prob
from torch.nn.utils import functional as F


def easy_display(root, k = 3, model_path = model_path, eos_id = eos_id):


	
	with open(os.path.join(root, "base_decode.txt"), 'r', encoding="utf8") as f:
		string = f.read()
	print(string)
	print('-' * 64)
	tokenizer = AutoTokenizer.from_pretrained(model_path)
	df_base = pd.read_csv(os.path.join(root, "base_token_probs.txt"), sep='\t', header=0)
	# display(df_base)
	df_aux = {
		"max_id": [],
		"probs": [],
		"cands": [],
		"eos_prob": [],
	}
	base_logits = torch.load(os.path.join(root, "base_logits.pt"), map_location=torch.device('cpu'))
	for tensor in base_logits:
		tensor_to_prob = F.softmax(tensor[0], dim=-1)
		top_k = torch.topk(tensor_to_prob, k = k)
		top_k_values = top_k.values
		top_k_indices = top_k.indices
		
		max_id = top_k_indices[0].item()
		probs = show_tensor(top_k_values)
		cand_ids = show_tensor(top_k_indices)
		cand_tokens = [tokenizer.decode(token_id) for token_id in top_k_indices]
		eos_prob = tensor_to_prob[eos_id].item()
		df_aux["max_id"].append(max_id)
		df_aux["probs"].append(probs)
		df_aux["cands"].append(cand_tokens)
		df_aux["eos_prob"].append(eos_prob)
		# print(max_id, probs, cand_ids, cand_tokens)
	df_aux = pd.DataFrame(df_aux, columns=["max_id", "probs", "cands", "eos_prob"])
	df_concat = pd.concat([df_base, df_aux], axis=1)
	display(df_concat)

# @param model_name_or_path: [Str]
# @param prompt: [Str]
# @param max_length: [Int]
# @param eos_id: [Int] default 151643 referes to <|endoftext|> of Qwen-xxx
# @param k: [Int] the number of top-k tokens to display
def base_pipeline(model_name_or_path,
				  prompt,
				  max_length,
				  eos_id = 151643,
				  k = 3,
				  ):
	model = AutoModelForCausalLM(model_name_or_path, trust_remote_code=True)
	tokenizer = AutoModelForCausalLM(model_name_or_path, trust_remote_code=True)
	device = "cuda" torch.cuda.is_available() else "cpu"
	generate_kwargs = {"do_sample": False, "top_k": 0, "top_p": 1., "num_beams": 1, "temperature": 1}

	# Hook
	@register_forward_hook_decorator([
		"model.layers[0].self_attn.q_proj",
		"model.layers[0].self_attn.k_proj",
	])
	def wrapped_generate(model, tokenizer, prompt, max_length, generate_kwargs, device):
		text, token_prob, logits = generate_token_prob(model, tokenizer, prompt, max_length, generate_kwargs, device)
		return text, token_prob, logits

	text, token_prob, logits = wrapped_generate(model, tokenizer, prompt, max_length, generate_kwargs, device)
	df_token_prob = pandas.DataFrame(token_prob, columns=["id", "token", "prob"])
			
	def _display_tensor(_tensor, _round):
		return list(map(lambda x: round(x, _round), _tensor.tolist()))
	df_display = {
		"max_id": [],
		"cand_tokens": [],
		"cand_probs": [],
		"eos_prob": [],
	}
	for tensor in logits:
		tensor_to_prob = F.softmax(tensor[0], dim=-1)
		top_k = torch.topk(tensor_to_prob, k = k)
		top_k_values = top_k.values
		top_k_indices = top_k.indices
		max_id = top_k_indices[0].item()
		probs = _display_tensor(top_k_values, 4)
		cand_ids = _display_tensor(top_k_indices, 4)
		cand_tokens = [tokenizer.decode(token_id) for token_id in top_k_indices]
		eos_prob = tensor_to_prob[eos_id].item()
		df_display["max_id"].append(max_id)
		df_display["probs"].append(probs)
		df_display["cands"].append(cand_tokens)
		df_display["eos_prob"].append(eos_prob)
	df_display = pd.DataFrame(df_display, columns=["max_id", "cand_tokens", "cand_probs", "eos_prob"])
	df_display = pd.concat([df_token_prob, df_display], axis=1)
	return df_display




	

def demo_1(mid=1):
	max_length = 4096
	device = "cuda" if torch.cuda.is_available() else "cpu"
	input_text = "请问英文单词strawberry中有几个字母r？"
	input_text = """请问英文单词strawberry中有几个字母r？ 要求：给出详细的思考过程。
嗯，我现在得仔细想想这个问题：英文单词strawberry中有几个字母r。这个问题看起来不复杂，但作为刚开始学习英语的人来说，可能会有点挑战性。好，我得一步一步来，确保自己没有遗漏任何细节。

首先，我需要明确问题。问题是在单词“strawberry”中有多少个字母“r”。那我得先把这个单词分解开来，逐个字母地检查，找出所有出现的字母“r”。

接下来，我应该把这个单词写出来，这样更直观"""
	input_text = """请问英文单词strawberry中有几个字母r？ 要求：给出详细的思考过程。
嗯，我现在得仔细想想这个问题：英文单词strawberry中有几个字母r。这个问题看起来不复杂，但作为刚开始学习英语的人来说，可能会有点挑战性。好，我得一步一步来，确保自己没有遗漏任何细节。

首先，我需要明确问题。问题是在单词“strawberry”中有多少个字母“r”。那我得先把这个单词分解开来，逐个字母地检查，找出所有出现的字母“r”。

接下来，我应该把这个单词写出来，这样更直观。让我写下“strawberry”：S-T-R-A-W-B-E-R-R-Y。哦，不对，这可能不对，因为有时候拼写可能会有错误。让我再仔细拼一遍。正确的拼写应该是S-T-R-A-W-B-E-R-R-Y吗？或者是不是中间有其他的字母？

等等，我觉得可能有问题。让我再确认一下“strawberry”的正确拼写。通常，strawberry是正确的，对吗？让我们数一下字母的数量：S, T, R, A, W, B, E, R, R, Y。所以总共有10个字母？不对，strawberry应该是有10个字母吗？让我数一遍：S(1), T(2), R(3), A(4), W(5), B(6), E(7), R(8), R(9), Y(10)。所以是的，是10个字母。

现在，我需要找出其中有多少个字母r。让我一个一个字母地检查。

第一个字母是S，不是r。
第二个字母是T，不是r。
第三个字母是R，是r，记下来，这是第一个r。
第四个字母是A，不是r。
第五个字母是W，不是r。
第六个字母是B，不是r。
第七个字母是E，不是r。
第八个字母是R，是r，这是第二个r。
第九个字母是R，又是r，这是第三个r。
第十个字母是Y，不是r。

等等，这好像不对，因为strawberry通常不会拼成三个r。让我再仔细检查一遍，可能我数错了字母的位置。

正确的拼写应该是S-T-R-A-W-B-E-R-R-Y吗？或者是不是中间少了一个r？让我再确认一下。strawberry的正确拼写应该是"""
	input_text = """让我再确认一下“strawberry”的正确拼写。通常，strawberry是正确的，对吗？让我们数一下字母的数量：S, T, R, A, W, B, E, R, R, Y。所以总共有10个字母？不对，strawberry应该是有10个字母吗？让我数一遍：S(1), T(2), R(3), A(4), W(5), B(6), E(7), R(8), R(9), Y(10)。所以是的，是10个字母。

现在，我需要找出其中有多少个字母r。让我一个一个字母地检查。

第一个字母是S，不是r。
第二个字母是T，不是r。
第三个字母是R，是r，记下来，这是第一个r。
第四个字母是A，不是r。
第五个字母是W，不是r。
第六个字母是B，不是r。
第七个字母是E，不是r。
第八个字母是R，是r，这是第二个r。
第九个字母是R，又是r，这是第三个r。
第十个字母是Y，不是r。

等等，这好像不对，因为strawberry通常不会拼成三个r。让我再仔细检查一遍，可能我数错了字母的位置。

正确的拼写应该是S-T-R-A-W-B-E-R-R-Y吗？或者是不是中间少了一个r？让我再确认一下。strawberry的正确拼写应该是S-T-R-A-W-B-E-R-R-Y，对吗？是的，应该是这样的，因为“straw”是S-T-R-A-W，然后加上“berry”是B-E-R-R-Y，所以中间的“berry”部分有两个r，对吗？

所以，整个单词中r出现的位置是第三个字母和第八个、第九个字母吗？不对，因为strawberry的正确结构应该是S-T-R-A-W-B-E-R-R-Y，对吗？让我再数一遍每个字母的位置：

1. S
2. T
3. R
4. A
5. W
6. B
7. E
8. R
9. R
10. Y

哦，对，所以从位置来看，第三个字母是R，第八个字母是R，第九个字母也是R。那是不是有三个r呢？这似乎有点多，因为通常strawberry中r的数量可能不是三个。

等等，这可能是个错误，因为有时候拼写可能会有变化，或者我的记忆有误。让我再确认一下。实际上，strawberry的正确拼写是S-T-R-A-W-B-E-R-R-Y，对吗？那确实有三个r吗"""
	input_text = """如何评价2024巴黎奥运会中王楚钦战胜张本智和？"""
	input_text = """如何评价2012伦敦奥运会乒乓球男单决赛中王皓战胜张继科夺冠？"""
	input_text = """请用pytorch实现transformers的多头注意力模块。"""	
	input_text = """求解不定积分：\( \int e^{x^2} dx \)"""
	input_text = """求解不定积分：\( \int e^{\frac{x^2}2} dx \)"""
	input_text = """颍川荀氏是三国历史上著名的门阀士族，其中又以荀彧的名声最大，那么你知道在颍川荀氏中，著名的烈女荀采与荀彧是什么血缘关系？"""
	input_text = """（问答题）颍川荀氏是三国历史上著名的门阀士族，其中又以荀彧的名声最大，那么你知道在颍川荀氏中，著名的烈女荀采与荀彧是什么血缘关系？"""
	input_text = """（选择题）颍川荀氏是三国历史上著名的门阀士族，其中又以荀彧的名声最大，那么你知道在颍川荀氏中，著名的烈女荀采与荀彧是什么血缘关系？"""
	input_text = """（）颍川荀氏是三国历史上著名的门阀士族，其中又以荀彧的名声最大，那么你知道在颍川荀氏中，著名的烈女荀采与荀彧是什么血缘关系？"""
	input_text = """颍川荀氏是三国历史上著名的门阀士族，其中又以荀彧的名声最大，那么你知道在颍川荀氏中，著名的烈女荀采与荀彧是什么血缘关系？<think>好的，我会直接输出内容，回答用户的问题，不需要思考。"""
	input_text = """请为中华人民共和国成立100周年书写一首七绝，要求：平仄韵脚"""
	input_text = """请为中华人民共和国成立100周年书写一首七律："""
	input_text = """请使用markdown语法编写一个3行4列的表格，表头为“姓名”、“年龄”、“性别”，剩余3行请随机构造3个人物的姓名、年龄以及性别填写。"""
	input_text = """请使用markdown语法编写一个4行4列的表格，表头为“姓名”、“年龄”、“性别”，剩余4行请随机构造4个人物的姓名、年龄以及性别填写。"""
	input_text = """请使用markdown语法编写一个4行4列的表格，表头为“姓名”、“年龄”、“性别”，剩余3行请随机构造4个人物的姓名、年龄以及性别填写。"""
	input_text = """请使用markdown语法编写一个4行4列的表格，表头为“姓名”、“年龄”、“性别”，剩余3行请随机构造4个人物的姓名、年龄以及性别填写。"""

	logging.info(f"Load model: {MODEL_PATHS[mid]}...")
	model = accelerate_load_model(
		MODEL_PATHS[mid], 
		Model = AutoModelForCausalLM,
		device_map = "auto",
		offload_folder = "./temp",
	)
	logging.info("  - ok!")
	tokenizer = AutoTokenizer.from_pretrained(MODEL_PATHS[mid])
	inputs = tokenizer.encode(input_text, return_tensors="pt").to(device)
	logging.info("Generate ...")
	outputs = model.generate(	# Greedy Decode Settings
		inputs,
		max_length = max_length,
		output_scores = True,
		return_dict_in_generate=True,
		do_sample = False,	# Disable sampling
		top_k = 0,			# Disable top-k
		top_p = 1.0,		# Disable top-p
		num_beams = 1,		# Disable beam search
		temperature = 1.0,	# Disable temperature
	)
	logging.info("  - ok!")
	generated_token_ids = outputs.sequences	# Long(1, max_length)
	generated_logits = outputs.scores	# Tuple(Float(1, n_vocab)) with length (max_length - n_tokens)
	generated_probs = tuple(map(lambda _logits: F.softmax(_logits, dim=-1), generated_logits))
	output_tokens = tokenizer.batch_decode(
		generated_token_ids,
		skip_special_tokens=True,
		clean_up_tokenization_spaces=False,
	)[0]
	diff = generated_token_ids.size(1) - len(generated_probs)
	token_probs = []
	for i in range(len(generated_probs)):
		token_id = generated_token_ids[0, i + diff].item()
		token = tokenizer.decode(token_id, skip_special_tokens=False)
		prob = generated_probs[i][0, token_id].item()
		token_probs.append((token_id, token, prob))
		
	token_probs_df = pandas.DataFrame(token_probs, columns=["id", "token", "prob"])
	token_probs_df.to_csv("./base_token_probs.txt", sep='\t', header=True, index=False)
	with open("./base_decode.txt", 'w', encoding="utf8") as f:
		f.write(output_tokens + '\n')
	torch.save(generated_logits, "base_logits.pt")
	
