import sys
import os
import torch
import torch_npu
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 指定 so 路径
custom_lib_path = "/root/wkv6/wkv6_Ascend/npu/build"

os.environ['WKV_LIB'] = custom_lib_path

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.rwkv6 import Rwkv6ForCausalLM, Rwkv6Tokenizer

def generate_prompt(instruction, input=""):
    instruction = instruction.strip().replace('\r\n','\n').replace('\n\n','\n')
    input = input.strip().replace('\r\n','\n').replace('\n\n','\n')
    if input:
        return f"""Instruction: {instruction}

Input: {input}

Response:"""
    else:
        return f"""User: hi

Assistant: Hi. I am your assistant and I will provide expert full response in full details. Please feel free to ask any question and I will always answer it.

User: {instruction}

Assistant:"""

device = 'npu:0'
model = Rwkv6ForCausalLM.from_pretrained("RWKV/rwkv-6-world-1b6", torch_dtype=torch.float16).to(device)
tokenizer = Rwkv6Tokenizer.from_pretrained("RWKV/rwkv-6-world-1b6")


text = "请介绍北京的旅游景点"
prompt = generate_prompt(text)

inputs = tokenizer(prompt, return_tensors="pt").to(device)
print("ready to decode!")
import time
start_time = time.time()
output = model.generate(inputs["input_ids"], max_new_tokens=200, do_sample=True, temperature=1.0, top_p=0.3, top_k=0, )


print(tokenizer.decode(output[0].tolist(), skip_special_tokens=True))
print(time.time() - start_time)
