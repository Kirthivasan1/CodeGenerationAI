
# CodeModel.py

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
import torch
import streamlit as st
import re

@st.cache_resource
def load_model():
    model_id = "codellama/CodeLlama-7b-Instruct-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=bnb_config,
        torch_dtype=torch.float16
    )

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"
    )
    return pipe, tokenizer

pipe, tokenizer = load_model()



def generate_code(user):

    def truncate_to_last_sentence(text):
        # Remove hanging bullet like "19." at end
        match = re.search(r"(.*?)(\n\d+\.\s*)?$", text.strip(), re.DOTALL)
        if match:
            text = match.group(1)

        # Now trim to last full sentence
        if '.' in text:
            last_period = text.rfind('.')
            return text[:last_period + 1].strip()
        return text.strip()


    # system = "You are CodeGenie, a concise programming tutor. Summarize what the code does, evaluate logic and style, then suggest clear fixes."
    system = "You are CodeGenie, a concise programming tutor. Summarize what the code does, evaluate logic and style, then suggest clear fixes."
    '''user = """Analyze this Python function and give feedback:

    def add(a, b):
      return a + b"""'''
    prompt = f"<s><<SYS>>\n{system}\n<</SYS>>\n\n{user}"

    prompt_tokens = tokenizer(prompt, return_tensors="pt").input_ids.shape[-1]
    max_total_tokens = 256
    max_new_tokens = max(32, max_total_tokens - prompt_tokens)

    sequences = pipe(
        prompt,
        do_sample=True,
        top_k=10,
        temperature=0.1,
        top_p=0.95,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=max_new_tokens,
        add_special_tokens=False
    )

    generated = sequences[0]['generated_text']
    if prompt in generated:
        result = generated.replace(prompt, "").strip()
    else:
        # In case model doesn't echo prompt, just return from the first new line
        result = generated.split('\n', 1)[-1].strip()

    result = truncate_to_last_sentence(result)

    return result