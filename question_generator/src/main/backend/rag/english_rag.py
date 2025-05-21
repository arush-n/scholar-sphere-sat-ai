#!/usr/bin/env python3
"""
english_rag.py

Enhanced SAT Reading question generator, tailored per category:
- Fetches robust background from Wikipedia, Wikidata, DBpedia
- Uses a random skeleton question from category JSON for structure
- Dynamically frames passage generation using skeleton paragraph structure
- Generates coherent 75–200 word passages with citation aside
- Varies passage prompt by category (e.g. transitions vs. theme)
- Builds an LLM-derived Knowledge Graph with dynamic nodes, edges, and properties
- Generates correct question + answer, then informed distractors based on KG
- Structured rationale with line references
- Style-example fallback
- Clean SAT-like output formatting and detailed logging
"""

import json
import random
from pathlib import Path
import requests
import torch
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import wikipedia
import spacy
import networkx as nx

# === Configuration ===
MODEL_REPO = "Qwen/Qwen2.5-7B-Instruct"
CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_MEMORY = {0: "15GiB"} if DEVICE.type == "cuda" else {"cpu": "100GiB"}
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_enable_fp32_cpu_offload=False
)
nlp = spacy.load("en_core_web_sm")

# Determine project root and question DB path
PROJECT_ROOT = Path(__file__).resolve().parents[5]
QUESTION_DB = PROJECT_ROOT / "question_synthesizer" / "src" / "question_data" / "Reading"

# Map question categories to parent and default length/Lexile
CATEGORY_PARENTS = {
    'cross-text-connections': 'Craft and Structure',
    'text-structure-and-purpose': 'Craft and Structure',
    'words-in-context': 'Craft and Structure',
    'rhetorical-synthesis': 'Expression of Ideas',
    'transitions': 'Expression of Ideas',
    'central-ideas-and-details': 'Information and Ideas',
    'command-of-evidence': 'Information and Ideas',
    'inferences': 'Information and Ideas',
    'boundaries': 'Standard English Conventions',
    'form--structure--and-sense': 'Standard English Conventions'
}
LENGTH_LEX = {
    'Standard English Conventions': (75,100,900),
    'Craft and Structure': (100,175,1100),
    'Expression of Ideas': (100,175,1100),
    'Information and Ideas': (125,200,1200)
}

# === Background fetchers ===
def fetch_wikipedia_context(topic, sentences=3):
    for variant in [topic, topic.replace('-', ' '), topic.title()]:
        try:
            page = wikipedia.page(variant, auto_suggest=True)
            summary = wikipedia.summary(page.title, sentences=sentences)
            print(f"[LOG] Fetched Wikipedia context for '{page.title}'.")
            return summary.strip(), page.title
        except Exception:
            continue
    print(f"[WARN] No Wikipedia context for '{topic}'.")
    return "", None


def fetch_wikidata_context(topic):
    url = "https://www.wikidata.org/w/api.php"
    params = {'action':'wbsearchentities','format':'json','language':'en','search':topic}
    try:
        r = requests.get(url, params=params, timeout=5).json()
        if 'search' in r and r['search']:
            ent = r['search'][0]['id']
            data = requests.get(f"https://www.wikidata.org/wiki/Special:EntityData/{ent}.json", timeout=5).json()
            desc = data['entities'][ent]['descriptions']['en']['value']
            print(f"[LOG] Fetched Wikidata context for '{topic}'.")
            return desc
    except Exception as e:
        print(f"[WARN] Wikidata fetch failed: {e}")
    return ""


def fetch_dbpedia_context(topic):
    key = topic.replace(' ', '_')
    url = f"http://api.dbpedia.org/data/{key}.json"
    headers = {'Accept':'application/json'}
    try:
        r = requests.get(url, headers=headers, timeout=5).json()
        subj = f"http://dbpedia.org/resource/{key}"
        res = r.get(subj, {})
        abstracts = res.get('http://dbpedia.org/ontology/abstract', [])
        for item in abstracts:
            if item.get('lang')=='en':
                print(f"[LOG] Fetched DBpedia context for '{topic}'.")
                return item.get('value','')
    except Exception as e:
        print(f"[WARN] DBpedia fetch failed: {e}")
    return ""

# === LLM JSON extraction helper ===
def extract_json(prompt, tok, mod, max_tokens=256):
    inp = tok(prompt, return_tensors="pt", truncation=True).to(DEVICE)
    with torch.no_grad():
        out = mod.generate(**inp, max_new_tokens=max_tokens, do_sample=False)
    text = tok.decode(out[0], skip_special_tokens=True)
    try:
        return json.loads(text)
    except Exception:
        print(f"[WARN] JSON parse failed: {text}")
        return []

# === Build LLM-derived Knowledge Graph ===
def build_enriched_kg(text, tok, mod, category):
    print(f"[LOG] Building LLM-derived KG for '{category}'...")
    graph_prompt = (
        "You are an LLM that extracts a knowledge graph from a passage."
        " Given the passage below, return a JSON object with:"
        " 'nodes': list of {id:string, type:string, properties:dict}," 
        " 'edges': list of {source:string, target:string, relation:string}." 
        f" Passage: '''{text}'''"
    )
    graph_data = extract_json(graph_prompt, tok, mod, max_tokens=512)
    G = nx.DiGraph()
    for n in graph_data.get('nodes', []):
        nid = n.get('id'); props = {k:v for k,v in n.items() if k!='id'}
        G.add_node(nid, **props)
    for e in graph_data.get('edges', []):
        G.add_edge(e.get('source'), e.get('target'), relation=e.get('relation'))
    print(f"[LOG] KG built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

# === Model loading ===
def load_model():
    print("[LOG] Loading model...")
    root = snapshot_download(MODEL_REPO, cache_dir=str(CACHE_DIR), local_files_only=True)
    tok = AutoTokenizer.from_pretrained(root, use_fast=True)
    mod = AutoModelForCausalLM.from_pretrained(
        root,
        device_map="auto",
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
        max_memory=MAX_MEMORY
    )
    mod.eval()
    print("[LOG] Model ready.")
    return tok, mod

# === Generate passage with skeleton structure ===
def generate_passage(source, topic, skeleton_paragraph, background, bg_title, min_w, max_w, lexile, category, tok, mod):
    # Extract skeleton frame: split into sentences and label broadly
    doc = nlp(skeleton_paragraph)
    sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    frame_lines = []
    roles = ['Introduction', 'Detail', 'Contrast', 'Conclusion']
    for i, sent in enumerate(sentences):
        role = roles[i] if i < len(roles) else f'Sentence{i+1}'
        frame_lines.append(f"{role}: '{sent}'")
    frame_descr = "; ".join(frame_lines)

    citation = f"[Citation: {bg_title}]" if bg_title else ""
    extra = "Emphasize varied transitional phrases and logical flow." if category=='transitions' else "Include thematic depth; metaphors optional."
    prompt = (
        "You are an expert SAT passage author."
        f"Use this paragraph structure frame (role: sentence): {frame_descr}."
        f"Background:{background}"
        f"Now write a single coherent paragraph ({min_w}-{max_w} words) at ~{lexile}L about '{topic}' from a {source}."
        f"Follow the frame roles order, use diverse sentences, include 5+ advanced words. {extra}"
    )
    print("[LOG] Generating passage based on dynamic skeleton frame...")
    inp = tok(prompt, return_tensors="pt", truncation=True).to(DEVICE)
    out = mod.generate(**inp, max_new_tokens=300, do_sample=True, temperature=0.8, top_p=0.9)
    p = tok.decode(out[0], skip_special_tokens=True)
    print(f"CITATION ASIDE: {citation} PASSAGE: {p}")
    return p

# === Generate question & answer ===
def generate_question_and_answer(category, skeleton, passage, kg, tok, mod):
    ref = 'lines 1–2' if category!='transitions' else 'sentence connectors'
    prompt = (
        "You are an SAT question writer.\n"
        f"Skeleton Prompt: {skeleton['Prompt']}\n"
        f"Passage: '''{passage}'''\n"
        f"KG Nodes & Edges: {list(kg.nodes(data=True))}, {[(u,v,d['relation']) for u,v,d in kg.edges(data=True)]}.\n"
        f"Write a {category} question referencing {ref}.\n"
        "Label 'Question:', list A)–D), then 'Answer: X'."
    )
    print("[LOG] Generating question+answer...")
    inp = tok(prompt, return_tensors="pt", truncation=True).to(DEVICE)
    out = mod.generate(**inp, max_new_tokens=200, do_sample=True, temperature=0.7, top_p=0.9)
    qa = tok.decode(out[0], skip_special_tokens=True)
    print(f"QUESTION+ANSWER:\n{qa}")
    return qa

# === Generate distractors ===
def generate_distractors(sol_kg, category, tok, mod):
    rels = [(u,v,d['relation']) for u,v,d in sol_kg.edges(data=True)]
    prompt = (
        "You are an SAT test designer.\n"
        f"Category: {category}.\nKG Nodes & Edges: {list(sol_kg.nodes(data=True))}, {rels}.\n"
        "Generate exactly 3 incorrect options (<15 words) reflecting misinterpretations based on the KG."
    )
    print("[LOG] Generating distractors...")
    inp = tok(prompt, return_tensors="pt", truncation=True).to(DEVICE)
    out = mod.generate(**inp, max_new_tokens=200, do_sample=True, temperature=0.7, top_p=0.9)
    d = tok.decode(out[0], skip_special_tokens=True)
    print(f"DISTRACTORS:\n{d}")
    return d

# === Generate rationale ===
def generate_rationale(qa, passage, tok, mod):
    prompt = (
        "You are an SAT teacher.\n"
        f"Q+A: '''{qa}'''\n"
        f"Passage: '''{passage}'''\n"
        "Write rationale:\n1. Why the correct answer is correct (cite lines).\n2. Why the wrong choices are wrong (cite lines)."
    )
    print("[LOG] Generating rationale...")
    inp = tok(prompt, return_tensors="pt", truncation=True).to(DEVICE)
    out = mod.generate(**inp, max_new_tokens=300, do_sample=True)
    r = tok.decode(out[0], skip_special_tokens=True)
    print(f"RATIONALE:\n{r}")
    return r

# === Main pipeline ===
def main():
    tok, mod = load_model()
    src = input("Source (research paper, novel, report): ")
    topic = input("Topic: ")
    print("Categories:", ', '.join(CATEGORY_PARENTS.keys()))
    category = input("Choose reading category: ").strip()
    if category not in CATEGORY_PARENTS:
        print("Invalid category."); return

    parent = CATEGORY_PARENTS[category]
    files = list((QUESTION_DB/parent).rglob(f"{category}.json"))
    if not files:
        print(f"[ERROR] No JSON skeleton for category '{category}'. Exiting."); return
    data = json.load(open(random.choice(files), encoding="utf-8"))
    skeleton = data if isinstance(data, dict) else random.choice(data)

    min_w, max_w, lex_default = LENGTH_LEX[parent]
    lex_input = input(f"Enter Lexile level or press enter for {lex_default}: ")
    lexile = int(lex_input) if lex_input.isdigit() else lex_default

    bg, bg_title = fetch_wikipedia_context(topic)
    wd = fetch_wikidata_context(topic)
    db = fetch_dbpedia_context(topic)
    combined_bg = "\n".join(filter(None, [bg, wd, db]))

    # Generate passage with skeleton frame
    passage = generate_passage(src, topic, skeleton.get('Paragraph', ''), combined_bg, bg_title, min_w, max_w, lexile, category, tok, mod)

    sol_kg = build_enriched_kg(passage, tok, mod, category)
    qa = generate_question_and_answer(category, skeleton, passage, sol_kg, tok, mod)
    _ = generate_distractors(sol_kg, category, tok, mod)
    _ = generate_rationale(qa, passage, tok, mod)

if __name__ == "__main__":
    main()
