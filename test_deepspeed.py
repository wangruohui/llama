
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from time import perf_counter
import numpy as np
import transformers
from transformers.models.llama import LlamaTokenizer
# hide generation warnings
transformers.logging.set_verbosity_error()

torch.set_default_device("cuda")

def measure_latency(model, tokenizer, payload, generation_args, device):
    input_ids = tokenizer(payload, return_tensors="pt").input_ids.to(device)
    latencies = []
    # warm up
    for _ in range(2):
        _ =  model.generate(input_ids, **generation_args)
    # Timed run
    for _ in range(10):
        start_time = perf_counter()
        _ = model.generate(input_ids, **generation_args)
        latency = perf_counter() - start_time
        latencies.append(latency)
    # Compute run statistics
    time_avg_ms = 1000 * np.mean(latencies)
    time_std_ms = 1000 * np.std(latencies)
    time_p95_ms = 1000 * np.percentile(latencies,95)
    return f"P95 latency (ms) - {time_p95_ms}; Average latency (ms) - {time_avg_ms:.2f} +\- {time_std_ms:.2f};", time_p95_ms

# Model Repository on huggingface.co
model_id = "decapoda-research/llama-7b-hf"

# load model and tokenizer
tokenizer = LlamaTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).cuda()

payload = "Hello my name is Philipp. I am getting in touch with you because i didn't get a response from you. What do I need to do to get my new card which I have requested 2 weeks ago? Please help me and answer this email in the next 7 days. Best regards and have a nice weekend but it"
input_ids = tokenizer(payload,return_tensors="pt").input_ids.to(model.device)
print(f"input payload: \n \n{payload}")
logits = model.generate(input_ids, do_sample=True, num_beams=1, min_length=128, max_new_tokens=128)
print(f"prediction: \n \n {tokenizer.decode(logits[0].tolist()[len(input_ids[0]):])}")

payload="Hello my name is Philipp. I am getting in touch with you because i didn't get a response from you. What do I need to do to get my new card which I have requested 2 weeks ago? Please help me and answer this email in the next 7 days. Best regards and have a nice weekend but it"*2
print(f'Payload sequence length is: {len(tokenizer(payload)["input_ids"])}')

# generation arguments
generation_args = dict(
  do_sample=False,
  num_beams=1,
  min_length=128,
  max_new_tokens=128
)
vanilla_results = measure_latency(model, tokenizer, payload, generation_args, model.device)

print(f"Vanilla model: {vanilla_results[0]}")



import deepspeed
# init deepspeed inference engine
ds_model = deepspeed.init_inference(
    model=model,      # Transformers models
    mp_size=1,        # Number of GPU
    dtype=torch.float16, # dtype of the weights (fp16)
    replace_method="auto", # Lets DS autmatically identify the layer to replace
    replace_with_kernel_inject=True, # replace the model with the kernel injector
)
print(f"model is loaded on device {ds_model.module.device}")

from deepspeed.ops.transformer.inference import DeepSpeedTransformerInference
# assert isinstance(ds_model.module.transformer.h[0], DeepSpeedTransformerInference) == True, "Model not sucessfully initalized"

# Test model
example = "My name is Philipp and I"
input_ids = tokenizer(example,return_tensors="pt").input_ids.to(model.device)
logits = ds_model.generate(input_ids, do_sample=True, max_length=100)
tokenizer.decode(logits[0].tolist())

payload = (
    "Hello my name is Philipp. I am getting in touch with you because i didn't get a response from you. What do I need to do to get my new card which I have requested 2 weeks ago? Please help me and answer this email in the next 7 days. Best regards and have a nice weekend but it"
    * 2
)
print(f'Payload sequence length is: {len(tokenizer(payload)["input_ids"])}')

# generation arguments
generation_args = dict(do_sample=False, num_beams=1, min_length=128, max_new_tokens=128)
ds_results = measure_latency(ds_model, tokenizer, payload, generation_args, ds_model.module.device)

print(f"DeepSpeed model: {ds_results[0]}")


import torch
from torch.profiler import ProfilerActivity, profile, record_function
torch.cuda.synchronize()
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], with_stack=True
) as prof:
    logits = ds_model.generate(input_ids, do_sample=True, max_length=10)
torch.cuda.synchronize()

prof.export_chrome_trace("deepspeed.json")