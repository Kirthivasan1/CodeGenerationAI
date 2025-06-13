
import os

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_in_model
import torch
import time
from huggingface_hub import login
from dotenv import load_dotenv

load_dotenv()
token = os.getenv("HF_TOKEN")
login(token)


def generate_code(prompt):
    model_id = "Salesforce/codegen-350M-multi"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load model with disk offloading
    with init_empty_weights():
        model = AutoModelForCausalLM.from_pretrained(model_id)
    device_map = infer_auto_device_map(model, no_split_module_classes=["LlamaDecoderLayer"],
                                       max_memory={"cpu": "48GiB"})
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map=device_map,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map=device_map,
        torch_dtype=torch.float16
    )

    start = time.time()
    sequences = pipe(
        prompt,
        do_sample=True,
        top_k=10,
        temperature=0.1,
        top_p=0.95,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_length=128
    )
    result = ''.join([s['generated_text'] for s in sequences])
    print(result)
    end = time.time()
    return result, start, end
