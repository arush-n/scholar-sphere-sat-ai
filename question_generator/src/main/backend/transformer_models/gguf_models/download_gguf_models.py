#!/usr/bin/env python3
"""
download_and_test_safetensors.py

1) Downloads Qwen2.5-7B-Instruct and Qwen2.5-Math-7B-Instruct safetensors into your HF cache.
2) Prints out exactly where each landed.
3) Loads each in 8-bit via BitsAndBytesConfig + auto device_map (GPU only),
   then does a tiny generate to verify.
"""

import sys
from pathlib import Path
import torch
from huggingface_hub import snapshot_download
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

# 1) The two repos
MODEL_REPOS = {
    "reading": "Qwen/Qwen2.5-7B-Instruct",
    "math":    "Qwen/Qwen2.5-Math-7B-Instruct",
}

# 2) Cache dir
CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 3) Device and memory mapping
if torch.cuda.is_available():
    DEVICE = torch.device("cuda", 0)
    MAX_MEMORY = {0: "15GiB"}  # adjust as needed
else:
    DEVICE = torch.device("cpu")
    MAX_MEMORY = {"cpu": "100GiB"}
print(f"\n→ Using device: {DEVICE}")

# 4) 8-bit quant config (no CPU offload)
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_enable_fp32_cpu_offload=False,
)

def download_models() -> dict[str, Path]:
    downloaded = {}
    for name, repo in MODEL_REPOS.items():
        print(f"\n→ Downloading '{name}' from '{repo}' …")
        root = snapshot_download(
            repo_id=repo,
            cache_dir=str(CACHE_DIR),
            resume_download=True,
            local_files_only=False
        )
        downloaded[name] = Path(root)
        print(f"✔️  '{name}' snapshot at:\n   {root}")
    return downloaded

def test_models(models: dict[str, Path]):
    print("\n=== Verifying load & generate with Transformers 8-bit ===")
    for name, repo in MODEL_REPOS.items():
        print(f"\n— Testing '{name}' …")
        try:
            tok = AutoTokenizer.from_pretrained(
                repo,
                cache_dir=str(CACHE_DIR),
                use_fast=True,
                local_files_only=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                repo,
                cache_dir=str(CACHE_DIR),
                device_map="auto",
                max_memory=MAX_MEMORY,
                quantization_config=bnb_config,
                trust_remote_code=True,
                local_files_only=True,
            )
            # quick smoke test
            prompt = "Hello, world!"
            inputs = tok(prompt, return_tensors="pt").to(DEVICE)
            outputs = model.generate(**inputs, max_new_tokens=5)
            text = tok.batch_decode(outputs, skip_special_tokens=True)[0]
            print(f"  ✅ `{name}` generated: {text!r}")
        except Exception as e:
            print(f"  ❌ `{name}` failed: {e}")
            sys.exit(1)
    print("\n🎉 Models downloaded, quantized, and tested successfully on", DEVICE)

def main():
    models = download_models()
    print("\n=== Download Summary ===")
    for name, path in models.items():
        print(f"• {name:8} → {path}")
    test_models(models)

if __name__ == "__main__":
    main()
