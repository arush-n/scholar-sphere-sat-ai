#!/usr/bin/env python3
"""
rag_sat_pipeline.py

End-to-end hybrid RAG SAT item generator supporting reading passages and math problems,
with knowledge graph building, question generation, distractors, and rationales.
"""

import os
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

# Select device: GPU if available
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_MEMORY = {0: "15GiB"} if DEVICE.type == "cuda" else {"cpu": "100GiB"}

# 8-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_enable_fp32_cpu_offload=False,
)

# === Retriever class ===
class Retriever:
    def __init__(self, docs, embed_model="all-MiniLM-L6-v2"):  # GPU-backed
        """
        docs: list of dicts with 'text' and 'citation' keys
        """
        self.docs = docs
        self.embedder = SentenceTransformer(embed_model, device=DEVICE)
        texts = [d["text"] for d in docs]
        embeddings = self.embedder.encode(texts, convert_to_tensor=True)
        self.embeddings = embeddings.cpu().numpy()
        d = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(self.embeddings)

    def search(self, query, top_k=5):
        q_emb = self.embedder.encode([query], convert_to_tensor=True).cpu().numpy()
        D, I = self.index.search(q_emb, top_k)
        results, citations = [], []
        for idx in I[0]:
            results.append(self.docs[idx]["text"])
            citations.append(self.docs[idx].get("citation", f"[{idx}]") )
        return results, citations

# === Download and load models ===
def download_models():
    models = {}
    for name, repo in MODEL_REPOS.items():
        root = snapshot_download(
            repo_id=repo,
            cache_dir=str(CACHE_DIR),
            resume_download=True
        )
        tokenizer = AutoTokenizer.from_pretrained(root, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            root,
            device_map="auto",  # loads onto GPU if available
            max_memory=MAX_MEMORY,
            quantization_config=bnb_config,
            trust_remote_code=True
        )
        models[name] = (tokenizer, model)
    return models

# === Triple extraction for KG ===
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

# === KG Construction ===
def build_kg(text):
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    G = nx.DiGraph()
    for ent in doc.ents:
        G.add_node(ent.text, label=ent.label_)
    for s, p, o in extract_triples(doc):
        G.add_node(s)
        G.add_node(o)
        G.add_edge(s, o, relation=p)
    return G

# === Generation steps ===
def generate_reading_passage(query, retriever, model_pair):
    tokenizer, model = model_pair
    docs, citations = retriever.search(query)
    prompt = (
        "You are an SAT-style reading-passage author. "
        f"Given these snippets with citations {citations}, synthesize a ~100-word STEM passage."
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    passage = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return passage, citations


def generate_math_formula(topic, retriever, model_pair):
    tokenizer, model = model_pair
    examples, _ = retriever.search(topic)
    prompt = (
        "You are an SAT math problem author. "
        f"Based on these LaTeX examples {examples}, craft one medium-hard algebra problem in LaTeX."
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    formula = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return formula


def draft_question(context, kg, model_pair):
    tokenizer, model = model_pair
    prompt = (
        "You are a professional SAT question writer. "
        f"Using the context: {context} and KG nodes {list(kg.nodes)}, create a MCQ with 4 options."
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outs = model.generate(**inputs, max_new_tokens=200, do_sample=True)
    return tokenizer.decode(outs[0], skip_special_tokens=True)


def refine_question(draft, model_pair):
    tokenizer, model = model_pair
    prompt = (
        "You are an expert test-designer. Polish this draft for clarity and SAT style: "
        f"{draft}"
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outs = model.generate(**inputs, max_new_tokens=150, do_sample=False)
    return tokenizer.decode(outs[0], skip_special_tokens=True)


def solve_correct_answer(question, kg, model_pair):
    tokenizer, model = model_pair
    prompt = f"You are a student solving: {question}. Show steps and give final answer label."
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outs = model.generate(**inputs, max_new_tokens=200)
    return tokenizer.decode(outs[0], skip_special_tokens=True)


def generate_distractor(correct_solution, strategy, kg, model_pair):
    tokenizer, model = model_pair
    prompt = (
        f"You are a student making a common error ({strategy}). "
        f"Starting from this solution: {correct_solution}, introduce one mistake and finish."
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outs = model.generate(**inputs, max_new_tokens=150, do_sample=True)
    return tokenizer.decode(outs[0], skip_special_tokens=True)


def generate_rationale(question, choice, kg, model_pair):
    tokenizer, model = model_pair
    prompt = (
        f"You are a teacher explaining why a student might choose {choice} for: {question}."
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    outs = model.generate(**inputs, max_new_tokens=100)
    return tokenizer.decode(outs[0], skip_special_tokens=True)

# === Pipeline orchestration ===
def main():
    # 1) Download models
    models = download_models()

    # 2) Prepare retrievers (replace docs with your real corpora)
    reading_docs = [
        {"text": "Photosynthesis converts light...", "citation": "[1]"},
        {"text": "Chlorophyll is a pigment...", "citation": "[2]"},
    ]
    math_docs = [
        {"text": "Solve x^2 - 5x + 6 = 0.", "citation": "[M1]"},
        {"text": "Find the slope of line through (1,2) and (3,8).", "citation": "[M2]"},
    ]
    reading_retriever = Retriever(reading_docs)
    math_retriever = Retriever(math_docs)

    # 3) Generate core content
    passage, citations = generate_reading_passage(
        "STEM topic", reading_retriever, models["reading"]
    )
    formula = generate_math_formula(
        "quadratic equations", math_retriever, models["math"]
    )

    # 4) Build KG
    kg_text = passage if passage else formula
    kg = build_kg(kg_text)

    # 5) Question drafting & refinement
    draft = draft_question(
        passage or formula, kg, models["reading" if passage else "math"]
    )
    question = refine_question(draft, models["reading"])

    # 6) Solve correct answer & distractors
    correct = solve_correct_answer(question, kg, models["math"])
    distr = [
        generate_distractor(correct, strat, kg, models["math"])
        for strat in ["Step-Omission", "Formula-Mixer", "Alternate-Valid"]
    ]

    # 7) Rationales
    rationales = {
        choice: generate_rationale(question, choice, kg, models["reading"])
        for choice in [correct] + distr
    }

    # 8) Assembly
    sat_item = {
        "passage": passage, "citations": citations,
        "formula": formula, "question": question,
        "choices": [correct] + distr, "rationales": rationales
    }
    print("Generated SAT item:", sat_item)

if __name__ == "__main__":
    main()
