#!/usr/bin/env python3
"""
download_7b_models.py

1) Downloads Qwen2.5-7B-Instruct and Qwen2.5-Math-7B-Instruct into your HF cache.
2) Prints out exactly where each landed.
3) Loads each in 8-bit via BitsAndBytesConfig + auto device_map,
   forcing all layers onto the GPU (no CPU offload), then does a tiny generate.
"""

import sys
from pathlib import Path
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# 1) Define the two repos to pull
MODEL_REPOS = {
    "reading": "Qwen/Qwen2.5-7B-Instruct",
    "math":    "Qwen/Qwen2.5-Math-7B-Instruct",
}

# 2) Cache directory
CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 3) Compute device (always GPU:0 if available)
if torch.cuda.is_available():
    DEVICE = torch.device("cuda", 0)
    # For max_memory mapping, use integer key for GPU 0
    MAX_MEMORY = {0: "15GiB"}
else:
    DEVICE = torch.device("cpu")
    MAX_MEMORY = {"cpu": "100GiB"}  # plenty for CPU fallback

print(f"\n→ Using device: {DEVICE}")

# 4) 8-bit quant config (no CPU offload)
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
)

def download_models() -> dict[str, Path]:
    downloaded: dict[str, Path] = {}
    for name, repo in MODEL_REPOS.items():
        print(f"\n→ Downloading '{name}' model from '{repo}' …")
        path = snapshot_download(
            repo_id=repo,
            cache_dir=str(CACHE_DIR),
            resume_download=True,
            local_files_only=False
        )
        downloaded[name] = Path(path)
        print(f"✔️  '{name}' at:\n   {path}")
    return downloaded

def test_models():
    print("\n=== Testing Qwen models with 8-bit + GPU-only device_map ===")
    for name, repo in MODEL_REPOS.items():
        print(f"\n— Testing '{name}' ({repo}) …")
        try:
            tok = AutoTokenizer.from_pretrained(repo, use_fast=True)
            model = AutoModelForCausalLM.from_pretrained(
                repo,
                quantization_config=bnb_config,
                device_map="auto",
                max_memory=MAX_MEMORY,
                trust_remote_code=True
            )
            prompt = "Hello, world!"
            inputs = tok(prompt, return_tensors="pt").to(DEVICE)
            out_ids = model.generate(**inputs, max_new_tokens=5)
            text = tok.batch_decode(out_ids, skip_special_tokens=True)[0]
            print(f"  ✅ `{name}` generated: {text!r}")
        except Exception as e:
            print(f"  ❌ `{name}` failed: {e}")
            sys.exit(1)

def main():
    models = download_models()
    print("\n=== Download Summary ===")
    for name, path in models.items():
        print(f"• {name:8} → {path}")
    test_models()
    print(f"\n🎉 All Qwen2.5 models downloaded and tested successfully on {DEVICE}!")

if __name__ == "__main__":
    main()
