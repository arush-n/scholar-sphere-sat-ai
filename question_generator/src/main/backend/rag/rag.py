#!/usr/bin/env python3
"""
rag_sat_pipeline.py

Interactive hybrid RAG SAT item generator supporting reading passages and math problems,
with knowledge graph building and interactive generation.
"""

import os
import json
from pathlib import Path
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import faiss
import spacy
import networkx as nx

# === Configuration ===
MODEL_REPOS = {
    "reading": "Qwen/Qwen2.5-7B-Instruct",
    "math":    "Qwen/Qwen2.5-Math-7B-Instruct",
}
CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_MEMORY = {0: "15GiB"} if DEVICE.type == "cuda" else {"cpu": "100GiB"}

# Quantization config
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_enable_fp32_cpu_offload=False,
)

# Question DB path (for structure reference)
QUESTION_DATA_DIR = Path(r"C:\Users\rush\OneDrive\Documents\GitHub\scholar-sphere-sat-ai\question_synthesizer\src\question_data")
nlp = spacy.load("en_core_web_sm")

# === Retriever for DB structure examples ===
class Retriever:
    def __init__(self, docs, embed_model="all-MiniLM-L6-v2"):
        self.docs = docs
        self.embedder = SentenceTransformer(embed_model, device=DEVICE)
        texts = [d.get("text", "") for d in docs]
        embeddings = self.embedder.encode(texts, convert_to_tensor=True)
        arr = embeddings.cpu().numpy()
        self.index = faiss.IndexFlatL2(arr.shape[1])
        self.index.add(arr)

    def search(self, query, top_k=3):
        q_emb = self.embedder.encode([query], convert_to_tensor=True).cpu().numpy()
        _, idxs = self.index.search(q_emb, top_k)
        return [self.docs[i] for i in idxs[0]]

# === Load question DB for structure ===
def load_question_docs():
    math_docs, reading_docs = [], []
    for kind, container in [("Math", math_docs), ("Reading", reading_docs)]:
        for path in (QUESTION_DATA_DIR / kind).rglob("*.json"):
            try:
                data = json.load(open(path, encoding="utf-8"))
            except Exception:
                continue
            entries = data if isinstance(data, list) else [data] if isinstance(data, dict) else []
            for obj in entries:
                if not isinstance(obj, dict): continue
                # store full object for structure
                container.append(obj)
    return reading_docs, math_docs

# === Model loading ===
def download_models():
    models = {}
    for name, repo in MODEL_REPOS.items():
        root = snapshot_download(repo_id=repo, cache_dir=str(CACHE_DIR), local_files_only=True)
        tokenizer = AutoTokenizer.from_pretrained(root, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            root,
            device_map="auto",
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
            max_memory=MAX_MEMORY
        )
        model.eval()
        models[name] = (tokenizer, model)
    return models

# === KG utilities ===
def extract_triples(doc):
    triples = []
    for sent in doc.sents:
        roots = [t for t in sent if t.dep_ == "ROOT"]
        if not roots: continue
        root = roots[0]
        subj = next((w for w in root.lefts if w.dep_ == "nsubj"), None)
        obj = next((w for w in root.rights if w.dep_ in ("dobj", "pobj")), None)
        if subj and obj:
            triples.append((subj.text, root.lemma_, obj.text))
    return triples


def build_kg(text):
    print("Building knowledge graph...")
    doc = nlp(text)
    G = nx.DiGraph()
    for ent in doc.ents:
        G.add_node(ent.text, label=ent.label_)
    for s, p, o in extract_triples(doc):
        G.add_edge(s, o, relation=p)
    print(f"KG has {len(G.nodes())} nodes and {len(G.edges())} edges.")
    return G

# === Generation functions ===
def generate_reading_passage(prompt_text, model_pair):
    tok, mod = model_pair
    prompt = (
        f"You are an SAT-style reading-passage author. \n"  
        f"Write a ~100-word STEM passage based on prompt: '{prompt_text}'."
    )
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    out = mod.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7, top_p=0.9)
    passage = tok.decode(out[0], skip_special_tokens=True)
    print(f"Generated passage: {passage}\n")
    return passage


def generate_math_problem(topic, model_pair):
    tok, mod = model_pair
    prompt = (
        f"You are an SAT math problem author. \n"
        f"Create a medium-hard algebra problem about '{topic}' in LaTeX."
    )
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    out = mod.generate(**inputs, max_new_tokens=200, do_sample=True, temperature=0.7, top_p=0.9)
    problem = tok.decode(out[0], skip_special_tokens=True)
    print(f"Generated problem: {problem}\n")
    return problem


def generate_question(structure_examples, kg, gen_model, refine_model):
    print("\n--- Question Generation ---")
    # extract structure prompts
    examples_prompts = [ex.get("Prompt") for ex in structure_examples]
    prompt = (
        f"You are a professional SAT question writer.\n"
        f"Following these example prompts: {examples_prompts}\n"
        f"and given KG context nodes: {list(kg.nodes())}, draft one MCQ with 4 options."
    )
    tok, mod = gen_model
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    out = mod.generate(**inputs, max_new_tokens=200, do_sample=True)
    draft = tok.decode(out[0], skip_special_tokens=True)
    print(f"Draft question: {draft}\n")
    tok2, mod2 = refine_model
    inputs2 = tok2(f"Polish this SAT question: {draft}", return_tensors="pt").to(DEVICE)
    out2 = mod2.generate(**inputs2, max_new_tokens=150, do_sample=True)
    question = tok2.decode(out2[0], skip_special_tokens=True)
    print(f"Final question: {question}\n")
    return question

# === Main ===
def main():
    models = download_models()
    reading_db, math_db = load_question_docs()
    reading_ret = Retriever(reading_db)
    math_ret = Retriever(math_db)

    qtype = input("Enter question type (reading/math): ").strip().lower()
    if qtype == "reading":
        passage_prompt = input("Enter passage prompt: ")
        passage = generate_reading_passage(passage_prompt, models["reading"])
        kg = build_kg(passage)
        structure = reading_ret.search(passage_prompt)  # reference examples
        question = generate_question(structure, kg, models["reading"], models["reading"])
    elif qtype == "math":
        topic = input("Enter math topic: ")
        problem = generate_math_problem(topic, models["math"])
        kg = build_kg(problem)
        structure = math_ret.search(topic)
        question = generate_question(structure, kg, models["math"], models["math"])
    else:
        print("Invalid type. Choose 'reading' or 'math'.")

if __name__ == "__main__":
    main()
