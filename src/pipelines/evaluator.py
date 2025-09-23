# -*- coding: utf8 -*-
# @author: caoyang
# @email: caoyang@stu.sufe.edu.cn
# Evaluator for CAUSAL_LM

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from trl import PPOTrainer, PPOConfig, set_seed
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score


def mean_token_accuracy(self, predictions, references):
	correct_tokens = 0
	total_tokens = 0
	for pred, ref in zip(predictions, references):
		min_len = min(len(pred), len(ref))
		if min_len == 0:
			continue       
		pred_tokens = pred[:min_len]
		ref_tokens = ref[:min_len]
		correct_tokens += sum(1 for p, r in zip(pred_tokens, ref_tokens) if p == r)
		total_tokens += min_len
	return correct_tokens / total_tokens if total_tokens > 0 else 0

class CausalLMEvaluator:
    def __init__(self, model_name, dataset_name):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )
        self.dataset = load_dataset(dataset_name)
        
    def mean_token_accuracy(self, predictions, references):
        correct_tokens = 0
        total_tokens = 0
        
        for pred, ref in zip(predictions, references):
            # 对齐长度，取较短的长度
            min_len = min(len(pred), len(ref))
            if min_len == 0:
                continue       
            pred_tokens = pred[:min_len]
            ref_tokens = ref[:min_len]
            correct_tokens += sum(1 for p, r in zip(pred_tokens, ref_tokens) if p == r)
            total_tokens += min_len
        
        return correct_tokens / total_tokens if total_tokens > 0 else 0
    
    def perplexity(self, texts):
        perplexities = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                ppl = torch.exp(loss).item()
                perplexities.append(ppl)
        return np.mean(perplexities)
    
    def generate_response(self, prompt, max_length=100):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=inputs["input_ids"].shape[1] + max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return response
    
    def evaluate(self, num_samples=100):
        print(f"正在评估 {self.model_name} 在 {self.dataset_name} 上的性能...")
        test_data = self.dataset['train'].select(range(min(num_samples, len(self.dataset['train']))))
        token_accuracies = []
        generated_responses = []
        reference_completions = []
        perplexity_scores = []
        for i, sample in enumerate(tqdm(test_data)):
            prompt = sample['prompt']
            reference = sample['completion']
            # Generate response
            generated = self.generate_response(prompt)
            generated_responses.append(generated)
            reference_completions.append(reference)
            # Mean token accuracy
            gen_tokens = self.tokenizer.encode(generated, add_special_tokens=False)
            ref_tokens = self.tokenizer.encode(reference, add_special_tokens=False)
            token_acc = self.mean_token_accuracy([gen_tokens], [ref_tokens])
            token_accuracies.append(token_acc)
            # Perplexity
            full_text = prompt + " " + reference
            ppl = self.perplexity([full_text])
            perplexity_scores.append(ppl)
            
            if i < 3:
                print(f"\n--- 样本 {i+1} ---")
                print(f"提示: {prompt}")
                print(f"参考回复: {reference}")
                print(f"生成回复: {generated}")
                print(f"Token准确率: {token_acc:.4f}")
                print(f"困惑度: {ppl:.4f}")
        
        metrics = {
            'mean_token_accuracy': np.mean(token_accuracies),
            'median_token_accuracy': np.median(token_accuracies),
            'perplexity_mean': np.mean(perplexity_scores),
            'perplexity_median': np.median(perplexity_scores),
            'exact_match_rate': sum(1 for gen, ref in zip(generated_responses, reference_completions) 
                                  if gen.strip() == ref.strip()) / len(generated_responses)
        }
        return metrics, generated_responses, reference_completions

# 使用示例
if __name__ == "__main__":
    # 初始化评估器
    evaluator = CausalLMEvaluator(
        model_name="Qwen2/Qwen2-0.5B-Instruct",
        dataset_name="trl-lib/tldr"
    )

    metrics, generated, references = evaluator.evaluate(num_samples=50)
    
    print("\n" + "="*50)
    print("评估结果汇总:")
    print("="*50)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\n前5个生成示例:")
    for i, (gen, ref) in enumerate(zip(generated[:5], references[:5])):
        print(f"\n--- 示例 {i+1} ---")
        print(f"参考: {ref}")
        print(f"生成: {gen}")
        print(f"匹配: {'✓' if gen.strip() == ref.strip() else '✗'}")
