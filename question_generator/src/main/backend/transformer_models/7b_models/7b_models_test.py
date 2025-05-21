#!/usr/bin/env python3
"""
7b_models_test.py

End‐to‐end SAT‐style smoke‐test for Qwen2.5‐Instruct and Qwen2.5‐Math‐Instruct:
  • Loads reading model in 8-bit and math model in 4-bit on GPU via accelerate’s device_map
  • Generates a short passage + multiple‐choice question for reading
  • Generates a medium‐hard math problem + solution for math
"""

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)

# 1) Model repo IDs
MODELS = {
    "reading": "Qwen/Qwen2.5-7B-Instruct",
    "math":    "Qwen/Qwen2.5-Math-7B-Instruct",
}

# 2) SAT‐style prompts
PROMPTS = {
    "reading": (
        "You are a professional SAT reading author.\n"
        "Write a ~150-word passage about monarch butterfly migration and habitat loss.\n"
        "Then draft one SAT multiple-choice question (stem + 4 choices) and indicate the correct letter.\n"
    ),
    "math": (
        "You are a professional SAT math author.\n"
        "Create a medium-hard algebra problem in LaTeX involving a quadratic equation.\n"
        "After the problem, provide the correct answer and a brief step-by-step solution.\n"
    ),
}

def main():
    # 3) Two quantization configs:
    #    - 8-bit for reading (fits comfortably in GPU)
    #    - 4-bit for math (to avoid any CPU offload)
    bnb8 = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
    )
    bnb4 = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    for name, repo in MODELS.items():
        print(f"\n→ Preparing '{name}' pipeline with GPU‐only quant…")

        # choose config by model type
        quant_cfg = bnb4 if name == "math" else bnb8

        # load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(repo, use_fast=True)

        # load model quantized & mapped entirely to GPU
        model = AutoModelForCausalLM.from_pretrained(
            repo,
            quantization_config=quant_cfg,
            device_map="auto",
            trust_remote_code=True,
        )

        # create pipeline (model already on GPU)
        gen = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            trust_remote_code=True,
        )

        # generate
        print(f"\n--- {name.upper()} GENERATION ---\n")
        out = gen(
            PROMPTS[name],
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            repetition_penalty=1.1,
        )[0]["generated_text"]

        # print only the generated portion
        continuation = out[len(PROMPTS[name]):].strip()
        print(continuation)
        print("\n" + "=" * 60)

    print("\n🎉 SAT‐style passage/question & math problem generation complete!")

if __name__ == "__main__":
    main()
