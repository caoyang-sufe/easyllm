# -*- coding: utf8 -*-
# @author: caoyang
# @email: caoyang@stu.sufe.edu.cn

from openai import OpenAI

# Initialize OpenAI Client
# @param api_key: [Str] default as "EMPTY"
# @param base_url: [Str] default as "http://localhost:8888/v1"
# @return Object of openai._base_client.SyncAPIClient
def initialize_client(api_key = "EMPTY", 
					  base_url = "http://localhost:6006/v1",
					  **kwargs,
					  ):
	client = OpenAI(api_key=api_key, base_url=base_url, **kwargs)	
	return client

# @param client: @return of `initialize_client`
# @param model_name_or_path: Str, e.g. "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
# @param system_prompt: Str
# @param user_prompt: Str
# @return response: List of chunked content
def easy_chat(client,
			  model_name_or_path,
			  system_prompt,
			  user_prompt,
			  ):
	chat_response = client.chat.completions.create(
		model = model_name_or_path,
		messages = [
			{"role": "system", "content": system_prompt},
			{"role": "user", "content": user_prompt},
		],
		stream=True,
	)
	response = list()
	for chunk in chat_response:
		content = chunk.choices[0].delta.content
		print(content, end=str())
		response.append(content)
	return response
