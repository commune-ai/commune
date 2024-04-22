from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from flask import Flask, request, jsonify
import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from typing import List, Dict
import commune as c


class BittensorServer(c.Module):

    num_gpus = torch.cuda.device_count()
    executor = ThreadPoolExecutor(max_workers=num_gpus + 1)
    num_gpus = torch.cuda.device_count()
    processors = [BittensorServer(device=i) for i in range(num_gpus)]
    cnt = 0

    def __init__(self, device):
        self.tokenizer = AutoTokenizer.from_pretrained('robertmyers/ltargon-22b')
        self.model = AutoModelForCausalLM.from_pretrained('robertmyers/ltargon-22b', torch_dtype=torch.float16)
        self.pipeline = pipeline(
            "text-generation", self.model, tokenizer=self.tokenizer,
            device=device, max_new_tokens=150, temperature=0.9, top_p=0.9, do_sample=True, pad_token_id=self.tokenizer.eos_token_id
        )
        
    def apply_system_prompts(self, prompt):
        system_prompts = [
            "You are a helpful assistant.",
            "Your responses should be as if they're being evaluated by a college professor.",
            "Aim for concise, clear, and accurate answers.",
            "Avoid colloquial language and slang.",
            "Use formal language and proper grammar.",
            "Ensure your information is factual and evidence-based.",
            "Answer directly to the question without deviating from the topic.",
            "Each answer should be structured and well-organized.",
            "Provide succinct explanations, avoiding verbosity.",
            "Cite relevant examples if necessary, but remain brief."
        ]
        return "\n".join(system_prompts) + "\n" + prompt

    def forward(self, history) -> str:
        history_with_prompts = self.apply_system_prompts(history)
        resp = self.pipeline(history_with_prompts)[0]['generated_text'].split(':')[-1].replace(str(history_with_prompts), "")
        return resp.strip()

    def process_request(history, processor):
        response = processor.forward(history)
        return response
    

    @classmethod
    def run(cls, port:int=2023):
        app = Flask(__name__)

        @app.route('/process', methods=['POST'])
        def handle_request():
            print("Request Received!")
            cls.cnt += 1
            processor = cls.processors[cls.cnt % cls.num_gpus]
            try:
                history = json.loads(request.data)
                future = cls.executor.submit(cls.process_request, history, processor)
                response = future.result()
            except Exception as e:
                print("Error:", e)
                response = {"error": str(e)}
            return jsonify(response=response)
        
        parser = argparse.ArgumentParser(description='Run RobertMyers server.')
        app.run(host='0.0.0.0', port=port)