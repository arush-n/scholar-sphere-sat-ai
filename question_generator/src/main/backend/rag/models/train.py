import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from pathlib import Path
import spacy
import json
import random
import re
import requests # For background fetchers
import wikipedia # For background fetchers
import networkx as nx # For Knowledge Graph
import sys # For modifying Python path
import os # For path joining
import datetime # For unique filenames

# === Attempt to import get_all_contexts_for_rag from user's local module ===
INFO_FORMATTER_MODULE_BASE_PATH = r'C:\Users\rush\OneDrive\Documents\GitHub\scholar-sphere-sat-ai\question_generator\src\main\backend\rag'
INFO_FORMATTER_PACKAGE_PATH = os.path.join(INFO_FORMATTER_MODULE_BASE_PATH, 'information_methods')

get_all_contexts_for_rag = None

try:
    if INFO_FORMATTER_MODULE_BASE_PATH not in sys.path:
        sys.path.insert(0, INFO_FORMATTER_MODULE_BASE_PATH)
    
    from information_methods.info_formatter import get_all_contexts_for_rag as imported_get_all_contexts
    get_all_contexts_for_rag = imported_get_all_contexts
    print(f"[LOG] Successfully imported 'get_all_contexts_for_rag' from: {INFO_FORMATTER_PACKAGE_PATH}")
except ImportError as e:
    print(f"[ERROR] Failed to import 'get_all_contexts_for_rag'. Path tried: '{INFO_FORMATTER_MODULE_BASE_PATH}' for package 'information_methods.info_formatter'. Error: {e}")
    print(f"[INFO] Ensure '{INFO_FORMATTER_MODULE_BASE_PATH}' is in sys.path and contains 'information_methods/info_formatter.py' with the function.")
    print("[WARN] Proceeding without external RAG context fetching. Passage generation will rely solely on LLM's internal knowledge or provided topic.")
except Exception as e:
    print(f"[ERROR] An unexpected error occurred during custom module import: {e}")
    print("[WARN] Proceeding without external RAG context fetching.")

if get_all_contexts_for_rag is None:
    def get_all_contexts_for_rag(topic): # Dummy function
        print(f"[WARN] Using DUMMY 'get_all_contexts_for_rag'. No external context will be fetched for topic: {topic}")
        return [{"source_api": "Dummy", "source_title": topic, "context_text": "No external background context available (dummy function)."}]


# === Configuration ===
MODEL_REPO = "Qwen/Qwen2.5-7B-Instruct"
CACHE_DIR = Path.home() / ".cache" / "huggingface" / "hub"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_MEMORY_CONFIG = {}
if DEVICE.type == "cuda":
    try:
        max_gpu_mem_str = "15GiB"
        MAX_MEMORY_CONFIG = {0: max_gpu_mem_str}
        print(f"[LOG] CUDA available. Setting MAX_MEMORY for GPU 0 to: {max_gpu_mem_str}")
    except Exception as e:
        print(f"[WARN] Could not get GPU memory properties, defaulting MAX_MEMORY for GPU 0 to 15GiB. Error: {e}")
        MAX_MEMORY_CONFIG = {0: "15GiB"}
else:
    MAX_MEMORY_CONFIG = {"cpu": "100GiB"}
    print(f"[LOG] CUDA not available. Using CPU. MAX_MEMORY (RAM) set to: {MAX_MEMORY_CONFIG['cpu']}")

bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
)
print(f"[LOG] BitsAndBytesConfig created: load_in_8bit={bnb_config.load_in_8bit}, threshold={bnb_config.llm_int8_threshold}")

try:
    nlp_spacy = spacy.load("en_core_web_sm")
    print("[LOG] spaCy 'en_core_web_sm' model loaded.")
except OSError:
    print("[WARN] spaCy 'en_core_web_sm' not found. Downloading...")
    spacy.cli.download("en_core_web_sm")
    nlp_spacy = spacy.load("en_core_web_sm")
    print("[LOG] spaCy 'en_core_web_sm' model downloaded and loaded.")

try:
    PROJECT_ROOT = Path(__file__).resolve().parents[6] 
except NameError:
    PROJECT_ROOT = Path.cwd()
    print(f"[WARN] __file__ not defined, using current working directory as project root: {PROJECT_ROOT}")

QUESTION_DB = PROJECT_ROOT / "question_synthesizer" / "question_data" / "Reading" 
GENERATED_QUESTIONS_DIR = PROJECT_ROOT / "question_generator" / "generated_sat_questions"
GENERATED_QUESTIONS_DIR.mkdir(parents=True, exist_ok=True) 

print(f"[LOG] Project Root (estimated): {PROJECT_ROOT}")
print(f"[LOG] Question DB Path (Reading skeletons): {QUESTION_DB}")
print(f"[LOG] Path for Storing Generated Questions: {GENERATED_QUESTIONS_DIR}")


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
# Adjusted LENGTH_LEX for 30-250 word passages where applicable
LENGTH_LEX = { 
    'Rhetorical Synthesis_Notes': (50, 150, 1000), 
    'Transitions_Context': (50, 150, 1000), 
    'Standard English Conventions': (30,100,900), 
    'Craft and Structure': (75,250,1100), # Adjusted min for thematic passages
    'Expression of Ideas': (75,250,1100), 
    'Information and Ideas': (75,250,1200)
}
NOTE_BASED_CATEGORIES = ['rhetorical-synthesis', 'transitions'] 

DIFFICULTY_TARGETS = {
    "Easy": {"success_rate": 0.85, "description": "should be answerable with straightforward comprehension and minimal complex inference. Correct answer is clearly supported; distractors are relatively easy to eliminate.", "correct_answer_guidance": "The correct answer should be a clear paraphrase or directly inferable from a specific part of the text. Avoid direct quotes. Ensure it is distinct from distractors.", "distractor_guidance": "Distractors should be clearly incorrect upon careful reading, perhaps touching on passage themes but not answering the question, or representing very common, simple misunderstandings."},
    "Medium": {"success_rate": 0.70, "description": "may require careful reading, integrating information, or making simple inferences. Correct answer requires some thought; distractors are somewhat plausible.", "correct_answer_guidance": "The correct answer may require synthesizing information from a few sentences or making a well-supported inference. It must NOT be a direct quote. It should be subtly distinct from the most tempting distractor.", "distractor_guidance": "Distractors should be plausible misinterpretations or focus on secondary details, requiring careful analysis to dismiss. One distractor could be a near-miss, differing from the correct answer by a key subtlety."},
    "Hard": {"success_rate": 0.50, "description": "demands deep understanding, complex inference, synthesis of multiple details, or nuanced interpretation. Correct answer is subtly supported; distractors should be very plausible and tempting.", "correct_answer_guidance": "The correct answer should require nuanced understanding, synthesis of multiple passage elements, or a complex inference. It must NOT be obvious and should be a sophisticated paraphrase. The distinction between it and the best distractor might be very fine.", "distractor_guidance": "Distractors should be highly tempting, perhaps by using passage language misleadingly, representing common complex misunderstandings based on the Passage KG, or being 'almost correct' but flawed in a critical, subtle way. Aim for distractors of similar length and syntactic complexity to the correct answer."}
}


model = None
tokenizer = None
try:
    print(f"[LOG] Loading tokenizer for {MODEL_REPO}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, cache_dir=CACHE_DIR, trust_remote_code=True)
    print(f"[LOG] Loading model {MODEL_REPO}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_REPO,
        quantization_config=bnb_config,
        device_map="auto", 
        max_memory=MAX_MEMORY_CONFIG,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float16 if DEVICE.type == "cuda" else torch.float32,
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    model.eval()
    print(f"[LOG] Model {MODEL_REPO} loaded and in eval mode.")
    if hasattr(model, 'hf_device_map'): print(f"[LOG] Model device map: {model.hf_device_map}")
    elif hasattr(model, 'device'): print(f"[LOG] Model loaded on device: {model.device}")
except Exception as e:
    print(f"[ERROR] Error loading model or tokenizer: {e}")
    model = None

PERSONAS = {
    "test_designer": "You are a professional test designer for the SAT. Your goal is to create high-quality, fair, and challenging questions and passages that accurately assess skills for 10th and 11th-grade students. Ensure your content is original, well-structured, and appropriate for this audience.",
    "student": "You are a 10th/11th-grade student trying to answer an SAT question. Think through the problem carefully, consider different interpretations, and explain your thought process for arriving at an answer, including potential mistakes or alternative paths.",
    "teacher": "You are an experienced SAT teacher. Your task is to explain the reasoning behind SAT questions and answers. Clearly articulate why the correct answer is right and why the incorrect answers are wrong, providing insightful explanations that would help a student understand the underlying concepts and common pitfalls.",
    "kg_extractor": "You are an expert system that extracts structured information from text to build a knowledge graph. Focus on key entities, concepts, claims, evidence, arguments, and literary devices (if applicable) that are crucial for deep comprehension and question generation. Strive for a comprehensive graph with 15-30 meaningful, granular nodes and 20-40 rich interconnections if text complexity allows. Provide output strictly in the requested JSON format: {'nodes': [{'id':'node_id_str', 'type':'entity_type_str', 'properties':{...}}], 'edges': [{'source':'node_id_1_str', 'target':'node_id_2_str', 'relation':'relationship_type_str'}]}. Ensure all node IDs in edges exist in nodes. Node IDs should be descriptive strings from the text."
}

def call_llm(prompt_text, persona_key="test_designer", temperature=0.7, max_new_tokens=512, do_sample=True):
    if not model or not tokenizer:
        print("[ERROR] LLM not loaded. Cannot generate text.")
        return "Error: LLM not available."

    full_prompt = f"<|im_start|>system\n{PERSONAS.get(persona_key, PERSONAS['test_designer'])}<|im_end|>\n<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
    
    target_device = DEVICE
    if hasattr(model, 'device') and model.device.type != 'cpu':
        target_device = model.device
    elif hasattr(model, 'hf_device_map') and '' in model.hf_device_map and isinstance(model.hf_device_map[''], int) and model.hf_device_map[''] == 0 : 
         target_device = torch.device(f"cuda:{model.hf_device_map['']}")

    max_prompt_len_for_model = tokenizer.model_max_length - max_new_tokens - 20 
    
    inputs = tokenizer(full_prompt, return_tensors="pt", truncation=True, max_length=max_prompt_len_for_model).to(target_device)

    try:
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 0.01,
                do_sample=do_sample,
                eos_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id else tokenizer.convert_tokens_to_ids("<|im_end|>"),
                pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id,
                top_p=0.9 if do_sample else None,
                top_k=None if not do_sample else 50, 
                num_beams=1 
            )
        generated_ids = outputs[0][inputs.input_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return generated_text.strip()
    except Exception as e:
        print(f"[ERROR] Error during LLM generation: {e}"); return f"Error during generation: {e}"

def extract_json_from_llm(prompt_text, persona_key="kg_extractor", max_tokens=1536): 
    raw_output = call_llm(prompt_text, persona_key=persona_key, temperature=0.1, max_new_tokens=max_tokens, do_sample=False)
    if "Error:" in raw_output: print(f"[ERROR] LLM call failed during JSON extraction: {raw_output}"); return {}
    json_str = None
    try:
        match_multiline = re.search(r"```json\s*({[\s\S]*?})\s*```", raw_output)
        if match_multiline: json_str = match_multiline.group(1)
        else:
            first_brace = raw_output.find('{'); last_brace = raw_output.rfind('}')
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace: json_str = raw_output[first_brace : last_brace+1]
            elif raw_output.strip().startswith("{") and raw_output.strip().endswith("}"): json_str = raw_output.strip()
        if json_str: return json.loads(json_str)
        else: print(f"[WARN] No clear JSON block found. Raw: {raw_output[:200]}..."); return {}
    except json.JSONDecodeError as e: print(f"[WARN] JSON parse failed. Error: {e}. String: '{json_str if json_str else raw_output[:200]}...'"); return {}


def build_kg_from_text(text_content, kg_purpose_description="general understanding", category=None):
    print(f"[LOG] Building LLM-derived KG from text for purpose: '{kg_purpose_description}'. Text length: {len(text_content)} chars.")
    if not text_content.strip(): print("[WARN] Empty text content for KG. Returning empty graph."); return nx.DiGraph()

    focus_prompt = "Focus on key entities (people, places, specific concepts, themes), their defining attributes from the text, and the relationships between them (e.g., cause-effect, comparison, evidence-for-claim, character-interaction, thematic-link, sequence, part-of, property-of)."
    if kg_purpose_description == "passage content analysis": 
        if category in NOTE_BASED_CATEGORIES:
            focus_prompt = "From these notes, extract individual statements/claims as nodes, their core meaning as properties, and any explicit or implicit connections (e.g., support, contrast, sequence, elaboration, cause) between them as edges. Aim for granular nodes representing each distinct piece of information and detailed, varied relationships."
        elif category: 
             focus_prompt = f"For this '{category}' passage, extract main ideas, supporting details, arguments, claims, counter-claims, evidence presented, character actions/motivations (if applicable), and significant literary or rhetorical devices as nodes. Identify relationships like 'supports', 'contradicts', 'exemplifies', 'leads to', 'is_defined_by', 'is_characterized_by'."
    elif kg_purpose_description.startswith("planning content on"): # Topic KG
        focus_prompt = "From this background information, extract core concepts, key figures/events, fundamental principles, and their primary relationships relevant to the topic. Aim for a foundational graph that can guide the creation of an original passage by highlighting essential elements and connections."


    graph_prompt = (
        f"Given the following text, extract a comprehensive and interconnected knowledge graph. This KG is intended for {kg_purpose_description}.\n"
        f"{focus_prompt}\n"
        f"Return a JSON object with two keys: 'nodes' and 'edges'.\n"
        f"'nodes': A list of dictionaries, where each dictionary is {{'id': 'descriptive_node_id_from_text', 'type': 'entity_type_e.g._Claim_Evidence_Person_Concept_Theme_LiteraryDevice', 'properties': {{'summary': 'brief_description_or_key_attribute_from_passage', 'text_reference': 'relevant_quote_or_phrase_from_passage_if_applicable'}} }}.\n"
        f"'edges': A list of dictionaries, where each dictionary is {{'source': 'source_node_id', 'target': 'target_node_id', 'relation': 'specific_relationship_description_e.g._supports_contradicts_elaborates_on_is_an_example_of_is_a_type_of'}}.\n"
        f"Strive for 15-30 meaningful, granular nodes and 20-40 rich, varied interconnections if the text complexity allows. Ensure all node IDs used in 'edges' are defined in 'nodes'. Node IDs should be unique and preferably human-readable strings derived directly from the text content.\n\n"
        f"Text Content:\n'''{text_content[:3000]}'''\n\n" 
        f"JSON Output:"
    )
    graph_data = extract_json_from_llm(graph_prompt, persona_key="kg_extractor", max_tokens=2048) 
    G = nx.DiGraph()
    nodes_data = graph_data.get('nodes', [])
    edges_data = graph_data.get('edges', [])
    if not nodes_data: print(f"[WARN] No nodes extracted for KG ({kg_purpose_description}). KG will be empty.")
    for node_info in nodes_data:
        node_id = node_info.get('id')
        if not node_id: continue
        attributes = {k: v for k, v in node_info.items() if k != 'id'}
        G.add_node(node_id, **attributes)
    for edge_info in edges_data:
        source_id, target_id, relation = edge_info.get('source'), edge_info.get('target'), edge_info.get('relation', 'related_to')
        if not source_id or not target_id: continue
        if source_id not in G: G.add_node(source_id, type="Unknown_Implicit_Source_From_Edge")
        if target_id not in G: G.add_node(target_id, type="Unknown_Implicit_Target_From_Edge")
        G.add_edge(source_id, target_id, relation=relation)
    print(f"[LOG] KG (for {kg_purpose_description}) built with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")
    return G

def generate_passage_or_notes(source_type, topic, skeleton_content, background_context, topic_kg, min_w, max_w, lexile, category, difficulty_str):
    introductory_phrase = f"Drawing insights from discussions on '{topic}'" 
    difficulty_details = DIFFICULTY_TARGETS.get(difficulty_str, DIFFICULTY_TARGETS["Medium"])
    
    if category in NOTE_BASED_CATEGORIES:
        print(f"[LOG] Generating 'student notes' style content for category: {category}")
        note_len_key = f"{category.replace('-', '_').capitalize()}_Notes"
        min_w, max_w, lexile_notes = LENGTH_LEX.get(note_len_key, (50, 150, lexile)) 

        notes_style_example = skeleton_content if skeleton_content else "Note 1: [Fact/Observation]. Note 2: [Related detail]. Note 3: [Contrasting or additional point]."
        
        topic_kg_summary = ""
        if topic_kg.number_of_nodes() > 0:
            nodes_summary = ", ".join(list(topic_kg.nodes)[:min(5, topic_kg.number_of_nodes())])
            topic_kg_summary = f"Key elements from background research (Topic KG) to consider for notes: {nodes_summary}..."

        prompt = (
            f"You are an SAT test designer creating content for a '{category}' question of '{difficulty_str}' difficulty.\n"
            f"Topic: '{topic}'.\n"
            f"Background Information (Use this to ensure factual plausibility and enrich the notes. If this background seems highly irrelevant to the core topic '{topic}', prioritize your internal knowledge about the topic, using the background only for minor, directly relevant details. The notes MUST align with the core topic. The complexity of information should reflect '{difficulty_str}' difficulty.):\n'''{background_context[:1000]}...'''\n"
            f"{topic_kg_summary}\n\n"
            f"Task: Generate a set of 3-5 concise student research notes on the given topic. These notes should contain related but distinct pieces of information that a student would need to synthesize or use for a '{category}' task. "
            f"The complexity and subtlety of the information in the notes should align with a '{difficulty_str}' difficulty SAT question ({difficulty_details['description']}).\n"
            f"The notes should be similar in style, complexity, and format to the following example (DO NOT COPY THE EXAMPLE CONTENT, only its structure and type of information):\n'''{notes_style_example}'''\n"
            f"Ensure the notes are factually plausible. Word count for all notes combined: {min_w}-{max_w} words. Strictly adhere to this word count.\n\n"
            f"Generated Student Notes:"
        )
        content_text = call_llm(prompt, persona_key="test_designer", temperature=0.6, max_new_tokens=max_w + 100)
        introductory_phrase = f"The following notes were taken by a student researching '{topic}':" 
    
    else: # Standard thematic passage generation
        print(f"[LOG] Generating thematic passage for category: {category}, difficulty: {difficulty_str}")
        doc = nlp_spacy(skeleton_content if skeleton_content else "This is an introductory sentence. This sentence provides more detail. This is a concluding sentence.")
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
        frame_lines = []
        roles = ['Introduction', 'Supporting Detail 1', 'Development/Example', 'Contrast/Further Detail', 'Supporting Detail 2', 'Conclusion']
        for i, sent_text in enumerate(sentences):
            role = roles[i] if i < len(roles) else f'Continuation Sentence {i - len(roles) + 1}'
            frame_lines.append(f"- {role}: (emulate structure of: '{sent_text}')")
        frame_description = "\n".join(frame_lines)

        if topic_kg.number_of_nodes() > 0:
            main_entities = [str(node) for node, data in topic_kg.nodes(data=True) if data.get('type') in ['Concept', 'Event', 'Person']][:2]
            if main_entities: introductory_phrase = f"Exploring themes related to {', '.join(main_entities)} within the context of '{topic}',"
        elif source_type: introductory_phrase = f"Reflecting the style of a {source_type} on '{topic}',"

        category_specific_instruction = "" 
        if category == 'transitions': category_specific_instruction = "Pay special attention to using varied and logical transitional phrases."
        else: category_specific_instruction = f"Develop the topic with thematic depth and clarity appropriate for the target difficulty ({difficulty_str}). Achieve complexity through ideas and sentence structure, not just obscure vocabulary."
        
        style_instruction = f"The passage should be written in the style of a {source_type}."
        if source_type.lower() == "poem": style_instruction = f"The passage should be in poetic form, with attention to meter, rhyme (optional), and figurative language, suitable for an SAT comprehension task. The complexity of language and themes should match a '{difficulty_str}' SAT question."
        # ... (other style instructions)

        topic_kg_summary = ""
        if topic_kg.number_of_nodes() > 0:
            nodes_summary = ", ".join(list(topic_kg.nodes)[:min(5, topic_kg.number_of_nodes())])
            topic_kg_summary = f"Key elements from background research (Topic KG) to guide content and ensure relevance: {nodes_summary}..."
        
        prompt = (
            f"You are an expert SAT passage author creating content for a '{category}' question of '{difficulty_str}' difficulty.\n"
            f"{introductory_phrase} the new passage should adhere to the following:\n"
            f"Topic: '{topic}'.\nStyle: {style_instruction}\n"
            f"Background Information & Topic KG Insights (Use this to ensure factual accuracy, incorporate key concepts from the Topic KG, and enrich details. If this background seems significantly off-topic from the primary user-defined `topic` of '{topic}' and `source_type` of '{source_type}', prioritize generating content based on your internal knowledge about the `topic` and `source_type`, using the background only for minor, directly relevant details. The generated passage MUST align with the user's core `topic` and `source_type`. Thematic complexity and reasoning required should match '{difficulty_str}' difficulty ({difficulty_details['description']}).):\n"
            f"Topic KG Summary: {topic_kg_summary}\nRaw Background (for cross-referencing if needed):\n'''{background_context[:1000]}...'''\n\n"
            f"Follow this structural frame. Adapt the content to the given topic and background, maintaining sentence roles and flow:\n{frame_description}\n\n"
            f"Instructions for the new passage:\n"
            f"- Write a single, coherent paragraph. Strictly adhere to word count: {min_w}-{max_w} words. Lexile Level: ~{lexile}L.\n"
            f"- {category_specific_instruction} Ensure originality and avoid plagiarism.\n\n"
            f"Generated Passage (beginning with the introductory phrase, or naturally incorporating the theme):"
        )
        content_text = call_llm(prompt, persona_key="test_designer", temperature=0.75, max_new_tokens=max_w + 200)
    
    print(f"[LOG] Generated Content Snippet ({'Notes' if category in NOTE_BASED_CATEGORIES else 'Passage'}): {content_text[:100]}...")
    return content_text, introductory_phrase


def generate_question_and_correct_answer_kg(category, skeleton_prompt_text, passage_or_notes, passage_kg, difficulty_str):
    kg_nodes_list = list(passage_kg.nodes(data=True))
    kg_edges_list = list(passage_kg.edges(data=True))
    kg_context_for_prompt = (f"Passage/Notes KG Nodes (sample): {json.dumps(kg_nodes_list[:min(3, len(kg_nodes_list))], indent=2)}\n"
                             f"Passage/Notes KG Edges (sample): {json.dumps(kg_edges_list[:min(3, len(kg_edges_list))], indent=2)}" 
                             if passage_kg.nodes() else "Passage/Notes KG: Not available or empty.")

    ref_question_style_hint = f"The style of the question should be similar to this skeleton prompt (adapt content to new passage/notes): '{skeleton_prompt_text}'"
    difficulty_guidance = DIFFICULTY_TARGETS.get(difficulty_str, DIFFICULTY_TARGETS["Medium"])

    line_ref_instruction = "The question or answer might implicitly refer to specific details or phrases without explicit line numbers."
    if category == 'command-of-evidence': 
        line_ref_instruction = "The question should ask for evidence. The correct answer must be directly and unequivocally supported by specific phrases or details in the passage/notes."
    elif category == 'rhetorical-synthesis':
        line_ref_instruction = "The question will likely ask the student to synthesize the provided notes to achieve a specific rhetorical goal. The correct answer will be the choice that best achieves this stated goal using the notes."
    
    prompt = (
        f"You are an SAT question writer specializing in '{category}' questions of '{difficulty_str}' difficulty (target success rate: ~{difficulty_guidance['success_rate']*100}%).\n"
        f"Provided Content (Passage or Student Notes):\n'''{passage_or_notes}'''\n\n"
        f"Knowledge Graph Insights from THIS Content (use these to identify complex relationships, key arguments, or subtle points to question):\n{kg_context_for_prompt}\n\n"
        f"Style Inspiration (from skeleton prompt for '{category}' questions):\n'''{skeleton_prompt_text}'''\n\n"
        f"Task:\n"
        f"1. Formulate a NEW SAT-style question based on the provided content and its KG insights. The question must match the '{category}' skill and be inspired by the skeleton prompt style. The question's complexity should align with '{difficulty_str}' difficulty: {difficulty_guidance['description']}\n"
        f"2. {line_ref_instruction}\n"
        f"3. Provide the text for the single BEST CORRECT answer choice. Crucially, this answer must NOT be a direct verbatim quote from the content; it should be a sophisticated paraphrase, require an inference, or synthesize information based on the content and its KG. It must be unequivocally supported. For '{difficulty_str}' difficulty, {difficulty_guidance['correct_answer_guidance']}\n\n"
        f"Output Format:\nQuestion: [Your new question text here]\nCorrect Answer Choice Text: [Text of the correct answer choice here]"
    )
    raw_output = call_llm(prompt, persona_key="test_designer", temperature=0.6, max_new_tokens=350) 
    question_text, correct_answer_text = "Error: Could not parse question.", "Error: Could not parse correct answer."
    q_match = re.search(r"Question:\s*([\s\S]*?)(Correct Answer Choice Text:|$)", raw_output, re.IGNORECASE)
    if q_match: question_text = q_match.group(1).strip()
    else:
        q_match_fallback = re.search(r"Question:\s*([\s\S]*)", raw_output, re.IGNORECASE)
        if q_match_fallback: question_text = q_match_fallback.group(1).strip()
    ca_match = re.search(r"Correct Answer Choice Text:\s*([\s\S]*)", raw_output, re.IGNORECASE)
    if ca_match: correct_answer_text = ca_match.group(1).strip()
    if "Error:" in correct_answer_text and "Error:" not in question_text and "\n" in raw_output:
        parts = raw_output.split("\n");
        for part in reversed(parts):
            if len(part.strip()) > 10 and not part.lower().startswith("question:"): correct_answer_text = part.strip(); break
    print(f"[LOG] Generated Question ({difficulty_str}): {question_text[:100]}...")
    print(f"[LOG] Generated Correct Answer Text ({difficulty_str}): {correct_answer_text[:100]}...")
    return question_text, correct_answer_text

def generate_distractors_kg(question_prompt, passage_or_notes, passage_kg, correct_answer_text, category, difficulty_str):
    kg_nodes_list = list(passage_kg.nodes(data=True))
    kg_edges_list = list(passage_kg.edges(data=True))
    kg_context_for_prompt = (f"Passage/Notes KG Nodes (sample): {json.dumps(kg_nodes_list[:3], indent=2)}\n"
                             f"Passage/Notes KG Edges (sample): {json.dumps(kg_edges_list[:3], indent=2)}" 
                             if passage_kg.nodes() else "Passage/Notes KG: Not available or empty.")
    difficulty_guidance = DIFFICULTY_TARGETS.get(difficulty_str, DIFFICULTY_TARGETS["Medium"])
    
    distractor_strategies_base = [ 
        "misrepresents a subtle relationship or detail from the Passage/Notes KG or content.",
        "focuses on a true but ultimately irrelevant entity/concept from the Passage/Notes KG/content in the context of the specific question asked.",
        "draws an overly broad, too narrow, or slightly flawed inference based on Passage/Notes KG connections or content details.",
        "confuses related but distinct concepts or terms found in the Passage/Notes KG or content.",
        "selects a detail that is factually correct according to the passage/notes but doesn't logically answer the question.",
        "offers an interpretation that is plausible on a superficial reading but is contradicted by deeper analysis of the passage/KG.",
        "twists the meaning of a key phrase or sentence from the passage/notes."
    ]
    if category == 'rhetorical-synthesis':
        distractor_strategies_base.extend([
            "only uses a subset of the relevant notes when a fuller synthesis is needed for the stated rhetorical goal.",
            "introduces an idea or detail not present in the provided notes, making it unsupported.",
            "synthesizes notes to achieve a different or incomplete rhetorical goal than the one specified in the question.",
            "correctly uses some notes but misinterprets the meaning or implication of one or more key notes, leading to a flawed conclusion."
        ])

    distractors = []
    current_choices_for_prompt = [correct_answer_text]
    for i in range(3):
        chosen_strategy = random.choice(distractor_strategies_base)
        prompt = (
            f"You are an SAT test designer creating challenging distractors for a '{category}' question of '{difficulty_str}' difficulty.\n"
            f"Content (Passage or Notes):\n'''{passage_or_notes}'''\nQuestion:\n'''{question_prompt}'''\nCorrect Answer Text:\n'''{correct_answer_text}'''\n"
            f"Knowledge Graph Insights from THIS Content (use to identify plausible but wrong paths/interpretations, or to select tempting but incorrect KG nodes/relationships for the distractor):\n{kg_context_for_prompt}\n"
            f"Existing Answer Choices (Correct + Prior Distractors):\n{json.dumps(current_choices_for_prompt)}\n\n"
            f"Task: Generate ONE new, unique, INCORRECT answer choice (distractor). It should be of similar length and syntactic complexity to the correct answer text provided.\n"
            f"The distractor must be a plausible misinterpretation. For '{difficulty_str}' difficulty ({difficulty_guidance['description']}), the distractor should be tempting by: {chosen_strategy} {difficulty_guidance['distractor_guidance']}\n"
            f"Ensure the distractor is related to the passage/notes content but definitively wrong upon careful analysis using the KG and text.\nNew Distractor Text:"
        )
        distractor_text = "Placeholder distractor (initial)"
        for attempt in range(3): 
            distractor_text = call_llm(prompt, persona_key="test_designer", temperature=0.75 + (attempt * 0.05), max_new_tokens=80) # Increased tokens for potentially longer, more complex distractors
            if "Error:" in distractor_text:
                if attempt == 2: distractor_text = f"Error placeholder distractor {i+1}"
                continue
            is_too_similar = any(abs(len(distractor_text) - len(existing)) < 5 and distractor_text[:10] == existing[:10] for existing in current_choices_for_prompt) 
            if distractor_text not in current_choices_for_prompt and len(distractor_text) > 5 and not is_too_similar: break
            if attempt == 2: distractor_text = f"Unique placeholder distractor {i+1} (attempts exhausted)"
        distractors.append(distractor_text)
        current_choices_for_prompt.append(distractor_text)
    return distractors

def generate_rationale_kg(question_prompt, passage_or_notes, answer_choices_obj, passage_kg, category, difficulty_str):
    # (generate_rationale_kg logic remains largely the same, adding difficulty context)
    choices_formatted = "\n".join([f"Choice {ac['label']}: \"{ac['text']}\"" for ac in answer_choices_obj["choices"]])
    correct_label = answer_choices_obj['correct_label']
    kg_nodes_list = list(passage_kg.nodes(data=True))
    kg_edges_list = list(passage_kg.edges(data=True))
    kg_context_for_prompt = (f"Passage/Notes KG Nodes (sample): {json.dumps(kg_nodes_list[:3], indent=2)}\n"
                             f"Passage/Notes KG Edges (sample): {json.dumps(kg_edges_list[:3], indent=2)}" 
                             if passage_kg.nodes() else "Passage/Notes KG: Not available or empty.")
    difficulty_guidance = DIFFICULTY_TARGETS.get(difficulty_str, DIFFICULTY_TARGETS["Medium"])

    analysis_persona_prompt = (
        f"First, as an analytical step (do not include this analysis in the final rationale output), consider why a student might choose each option for this '{difficulty_str}' question. "
        "For incorrect options, think about common misinterpretations of the passage/notes, or flawed reasoning paths based on the Passage/Notes KG that would be plausible for a student at this level. "
        "For the correct option, identify the key supporting evidence or logical steps."
    )
    prompt = (
        f"You are an expert SAT teacher explaining answers for a '{category}' question of '{difficulty_str}' difficulty.\n"
        f"Content (Passage or Notes):\n'''{passage_or_notes}'''\n\n"
        f"Question:\n'''{question_prompt}'''\n\nAnswer Choices:\n{choices_formatted}\nCorrect Answer: Choice {correct_label}\n"
        f"Knowledge Graph Insights from THIS Content (use to explain reasoning where relevant):\n{kg_context_for_prompt}\n\n"
        f"{analysis_persona_prompt}\n\n"
        f"Now, provide a detailed rationale for EACH answer choice (A, B, C, D), tailored for a student encountering a '{difficulty_str}' question ({difficulty_guidance['description']}).\n"
        f"- For CORRECT choice ({correct_label}), explain precisely why it is correct, citing specific textual evidence (phrases or sentences) from the content. Explain how Passage/Notes KG insights might confirm this.\n"
        f"- For EACH INCORRECT choice, explain its specific flaw, citing content evidence or Passage/Notes KG insights if helpful to show why it's wrong.\n"
        f"Format your response clearly, addressing each choice (A, B, C, D) separately.\n\n"
        f"Rationale:"
    )
    full_rationale_text = call_llm(prompt, persona_key="teacher", temperature=0.4, max_new_tokens=1024)
    # (Rationale parsing logic remains the same)
    rationales_dict = {}
    for choice_obj in answer_choices_obj["choices"]:
        label = choice_obj["label"]
        pattern = rf"(?s)(?:Rationale\s+for\s+)?Choice\s*{re.escape(label)}\s*[:\-–—]*\s*(.*?)(?=(?:Rationale\s+for\s+)?Choice\s*[A-D]\s*[:\-–—]|\Z)"
        match = re.search(pattern, full_rationale_text, re.IGNORECASE)
        if match: rationales_dict[label] = match.group(1).strip()
        else:
            rationales_dict[label] = f"Could not automatically extract rationale for Choice {label}."
            simple_find_start = full_rationale_text.upper().find(f"CHOICE {label}")
            if simple_find_start != -1:
                rough_text_after = full_rationale_text[simple_find_start:]
                next_choice_markers = [f"CHOICE {chr(ord('A')+k)}" for k in range(4) if chr(ord('A')+k) != label]
                ends = []
                for marker in next_choice_markers:
                    pos = rough_text_after.upper().find(marker, 10) 
                    if pos != -1: ends.append(pos)
                end_pos = min(ends) if ends else len(rough_text_after)
                rationales_dict[label] = rough_text_after[:end_pos].strip()
    return rationales_dict

def main_interactive_generator():
    if not model or not tokenizer:
        print("[FATAL] Model or Tokenizer not loaded. Cannot proceed with generation.")
        return

    print("\n=== SAT Reading Question Generator (Interactive Mode with Dual KG & Difficulty Tuning) ===")
    
    source_type = input("Enter source type for passage/notes inspiration (e.g., 'student research notes', 'historical document', 'literary criticism', 'novel excerpt', 'poem'): ").strip()
    if not source_type: source_type = "general academic text"

    topic = input("Enter the specific topic for the content: ").strip()
    if not topic: print("[ERROR] Topic cannot be empty. Exiting."); return

    difficulty_options = list(DIFFICULTY_TARGETS.keys())
    difficulty_str = input(f"Enter difficulty ({'/'.join(difficulty_options)}) or press Enter for Medium: ").strip().capitalize()
    if difficulty_str not in difficulty_options: difficulty_str = "Medium"
    print(f"[LOG] Selected Difficulty: {difficulty_str} (Target success: {DIFFICULTY_TARGETS[difficulty_str]['success_rate']*100}%)")


    print("\nAvailable Reading Categories:")
    for cat_key in CATEGORY_PARENTS.keys(): print(f"- {cat_key} ({CATEGORY_PARENTS[cat_key]})")
    category = input("Choose a reading category from the list above: ").strip()

    if category not in CATEGORY_PARENTS: print(f"[ERROR] Invalid category '{category}'. Exiting."); return

    parent_domain = CATEGORY_PARENTS[category]
    
    length_key_suffix = "_Notes" if category in NOTE_BASED_CATEGORIES else ""
    length_key_primary = f"{category.replace('-', '_').capitalize()}{length_key_suffix}"
    min_w, max_w, lex_default = LENGTH_LEX.get(length_key_primary, LENGTH_LEX.get(parent_domain, (50,250,1100))) 
    
    lexile_input = input(f"Enter target Lexile level (e.g., {lex_default}) or press Enter for default ({lex_default}L for {parent_domain} - type '{category}'): ").strip()
    lexile = int(lexile_input) if lexile_input.isdigit() else lex_default

    # (Skeleton loading logic remains similar)
    target_category_dir = QUESTION_DB / parent_domain
    skeleton_file_path = target_category_dir / f"{category}.json" 
    if not skeleton_file_path.exists():
        alt_category_name_space = category.replace('-', ' ')
        skeleton_file_path_alt_space = target_category_dir / f"{alt_category_name_space}.json"
        if skeleton_file_path_alt_space.exists(): skeleton_file_path = skeleton_file_path_alt_space
        else:
            alt_category_name_hyphen = category.replace(' ', '-')
            skeleton_file_path_alt_hyphen = target_category_dir / f"{alt_category_name_hyphen}.json"
            if skeleton_file_path_alt_hyphen.exists(): skeleton_file_path = skeleton_file_path_alt_hyphen
            else: print(f"[ERROR] No skeleton JSON file found for category '{category}' in directory '{target_category_dir}'. Exiting."); return
    print(f"[LOG] Loading skeleton from: {skeleton_file_path}")
    try:
        with open(skeleton_file_path, 'r', encoding='utf-8') as f: skeleton_data_list = json.load(f) 
        if not isinstance(skeleton_data_list, list) or not skeleton_data_list: print(f"[ERROR] Skeleton file {skeleton_file_path} not a list or empty. Exiting."); return
        skeleton_item = random.choice(skeleton_data_list) 
        if not isinstance(skeleton_item, dict): print(f"[ERROR] Chosen skeleton item not a dict. Item: {skeleton_item}. Exiting."); return
        skeleton_content_for_passage = skeleton_item.get('Paragraph', skeleton_item.get('Stimulus', "Default skeleton content."))
        skeleton_prompt_for_question = skeleton_item.get('Prompt', "Default skeleton prompt.")
        print(f"[LOG] Loaded skeleton for '{category}'. Prompt example: '{skeleton_prompt_for_question[:50]}...'")
    except Exception as e: print(f"[ERROR] Failed to load or parse skeleton JSON from {skeleton_file_path}: {e}. Exiting."); return

    print(f"\n[LOG] Fetching background context for topic: '{topic}'...")
    rag_results = get_all_contexts_for_rag(topic) 
    # (RAG result processing remains similar)
    combined_background_context, background_title = "No specific background context retrieved.", topic
    if rag_results and isinstance(rag_results, list):
        context_texts = [item.get("context_text", "") for item in rag_results if isinstance(item, dict) and item.get("context_text")]
        if context_texts: combined_background_context = "\n\n---\n\n".join(context_texts)
        # (Title selection logic remains similar)
        for item in rag_results:
            if isinstance(item, dict) and item.get("source_api") == "Wikipedia" and item.get("source_title"): background_title = item["source_title"]; break 
        if background_title == topic and context_texts: 
            for item in rag_results:
                if isinstance(item, dict) and item.get("source_title"): background_title = item["source_title"]; break 
    print(f"[LOG] Background context (Title: {background_title}, Length: {len(combined_background_context)} chars).")


    print("\n[LOG] Building Topic/Background Knowledge Graph...")
    topic_kg = build_kg_from_text(combined_background_context, f"planning content on '{topic}' for SAT category '{category}'", category=category)

    # Generate Passage or Notes based on category
    passage_or_notes_text, introductory_phrase = generate_passage_or_notes(
        source_type, topic, skeleton_content_for_passage, combined_background_context, topic_kg,
        min_w, max_w, lexile, category, difficulty_str
    )
    if "Error:" in passage_or_notes_text: print(f"[ERROR] Content generation failed: {passage_or_notes_text}"); return

    print("\n[LOG] Building Passage/Notes Knowledge Graph...")
    passage_notes_kg = build_kg_from_text(passage_or_notes_text, f"passage content analysis for SAT category '{category}'", category=category)

    question_text, correct_answer_text = generate_question_and_correct_answer_kg(
        category, skeleton_prompt_for_question, passage_or_notes_text, passage_notes_kg, difficulty_str
    )
    if "Error:" in question_text or "Error:" in correct_answer_text: print(f"[ERROR] Question or correct answer generation failed."); return

    distractor_texts = generate_distractors_kg(
        question_text, passage_or_notes_text, passage_notes_kg, correct_answer_text, category, difficulty_str
    )
    if not distractor_texts or any("Error:" in dt for dt in distractor_texts) or len(distractor_texts) != 3:
        print(f"[ERROR] Distractor generation failed. Using placeholders.")
        distractor_texts = [f"Placeholder Distractor {j+1}" for j in range(3)] if len(distractor_texts) !=3 else distractor_texts

    final_answer_choices_obj = {"choices": [], "correct_label": "A", "correct_text": correct_answer_text}
    final_answer_choices_obj["choices"].append({"label": "A", "text": correct_answer_text})
    for i, dt_text in enumerate(distractor_texts):
        final_answer_choices_obj["choices"].append({"label": chr(66 + i), "text": dt_text}) 
    
    print(f"\n[LOG] Final Answer Choices (Correct is A):")
    for ac in final_answer_choices_obj["choices"]: print(f"  {ac['label']}: {ac['text']}")
    print(f"[LOG] Correct Label: {final_answer_choices_obj['correct_label']}")

    rationales_dict = generate_rationale_kg(
        question_text, passage_or_notes_text, final_answer_choices_obj, passage_notes_kg, category, difficulty_str
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    safe_category_abbr = re.sub(r'\W+', '', category.replace('-', ''))[:3].upper()
    if not safe_category_abbr : safe_category_abbr = "GEN" 
    difficulty_abbr = difficulty_str[0].upper()
    safe_topic_fn = re.sub(r'\W+', '', topic.lower())[:15]
    
    question_id_str = f"SATR_{safe_category_abbr}{difficulty_abbr}_{timestamp}_{random.randint(100,999)}"
    filename = f"SAT_R_{safe_category_abbr}_{difficulty_abbr}_{safe_topic_fn}_{timestamp}.json"

    output_data = {
        "question_id": question_id_str,
        "assessment_type": "SAT Reading (Generated Interactive)",
        "category": category, "parent_domain": parent_domain, "topic": topic, "difficulty_level": difficulty_str,
        "source_type_inspiration": source_type, "target_lexile": lexile,
        "passage_introductory_phrase": introductory_phrase, 
        "passage_or_notes": passage_or_notes_text, 
        "topic_knowledge_graph_summary": {"nodes": topic_kg.number_of_nodes(), "edges": topic_kg.number_of_edges()},
        "passage_notes_knowledge_graph": { 
            "nodes": list(passage_notes_kg.nodes(data=True)), 
            "edges": list(passage_notes_kg.edges(data=True))
        },
        "question_prompt": question_text,
        "answer_choices": final_answer_choices_obj["choices"],
        "correct_answer_label": final_answer_choices_obj["correct_label"],
        "rationales": rationales_dict,
        "generation_metadata": { "model_repo": MODEL_REPO, "timestamp": timestamp, "filename": filename }
    }

    print("\n\n=== SAT Reading Question Generation Complete ===")
    output_json_str = json.dumps(output_data, indent=4, default=str) 
    # print(output_json_str) 

    try:
        output_file_path = GENERATED_QUESTIONS_DIR / filename
        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(output_json_str)
        print(f"\n[LOG] Generated question set saved to: {output_file_path}")
    except Exception as e:
        print(f"[ERROR] Error saving generated question to file: {e}")

    return output_data


if __name__ == "__main__":
    if model and tokenizer:
        main_interactive_generator()
    else:
        print("[FATAL] Model and/or tokenizer failed to load. Cannot start interactive generator.")

