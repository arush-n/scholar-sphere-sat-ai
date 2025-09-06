import json
import re
import spacy
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import os # Added for path operations
from pathlib import Path # Added for recursive file searching
import torch # Added for checking CUDA availability

# --- Configuration & Model Loading ---

# Check for CUDA availability and set device
device = 0 if torch.cuda.is_available() else -1
if device == 0:
    print(f"Device set to use cuda:{torch.cuda.current_device()}")
else:
    print("CUDA not available. Using CPU for Hugging Face models.")


# Load a spaCy model.
# You might need to download it first: python -m spacy download en_core_web_sm
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

# Load a zero-shot classification pipeline from Hugging Face Transformers
try:
    MODEL_NAME = "facebook/bart-large-mnli"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
    topic_classifier = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer, device=device)
    print(f"Zero-shot classification model '{MODEL_NAME}' loaded successfully.")
except Exception as e:
    print(f"Could not load zero-shot classification model. BERT metadata will be skipped. Error: {e}")
    topic_classifier = None

# Revised Candidate labels for zero-shot topic classification
CANDIDATE_TOPICS = [
    # Passage/Content Types
    "Literary Narrative", "Scientific Article Abstract", "Historical Document Analysis",
    "Social Science Commentary", "Poetry Analysis", "Informational Text Snippet", "Dialogue Excerpt",
    "Persuasive Writing Sample", "Comparative Texts",
    # Subject Areas (more granular)
    "Biology Concept", "Chemistry Principle", "Physics Application", "Earth Science Data",
    "US History Event", "World History Figure", "Economic Theory", "Psychological Study",
    "Sociological Observation", "Civics and Government", "Art History", "Music Analysis",
    # Specific Skills/Focus (complementing existing metadata)
    "Identifying Main Idea", "Analyzing Supporting Details", "Understanding Vocabulary in Context",
    "Determining Author's Purpose", "Assessing Author's Tone", "Evaluating Textual Evidence",
    "Making Inferences", "Analyzing Rhetorical Devices", "Interpreting Data from Graphs/Charts",
    "Logical Reasoning Problem", "Cross-Text Connection Analysis", "Theme Identification",
    # Math Specifics
    "Statistical Data Analysis", "Probability Calculation", "Algebraic Function Analysis",
    "Equation Solving Strategy", "Geometric Property Application", "Trigonometric Relationship",
    # Writing/Grammar Specifics
    "Sentence Structure Correction", "Punctuation Usage Rules", "Effective Word Choice (Diction)",
    "Logical Transitions", "Improving Clarity and Conciseness", "Maintaining Consistent Style"
]


# --- Helper Functions ---

def clean_text(text):
    """
    Cleans and normalizes a text string.
    """
    if not isinstance(text, str):
        return ""
    text = text.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"').replace("\u00a0", " ")
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def get_syntactic_parse_info(text):
    """
    Generates syntactic parse information for text using spaCy.
    """
    if not text:
        return []
    doc = nlp(text)
    sentences_info = []
    for sent in doc.sents:
        sentence_data = {"text": sent.text, "tokens": []}
        for token in sent:
            sentence_data["tokens"].append({
                "text": token.text, "lemma": token.lemma_, "pos": token.pos_,
                "tag": token.tag_, "dep": token.dep_, "head_text": token.head.text,
                "head_pos": token.head.pos_, "children": [child.text for child in token.children]
            })
        sentences_info.append(sentence_data)
    return sentences_info

def generate_bert_metadata(text_content):
    """
    Generates additional metadata using a zero-shot classification model.
    """
    if not topic_classifier or not text_content:
        return {"topics": [], "scores": [], "error": "Classifier not available or no text provided."}
    try:
        result = topic_classifier(text_content, CANDIDATE_TOPICS, multi_label=True)
        return {"topics": result['labels'], "scores": result['scores']}
    except Exception as e:
        print(f"Error during BERT metadata generation for text: '{text_content[:100]}...': {str(e)}")
        return {"topics": [], "scores": [], "error": f"Error during BERT metadata generation: {str(e)}"}


def preprocess_item(item_json, source_file_path="N/A"):
    """
    Preprocesses a single item (expected to be a dictionary) from the JSON data.
    Detects open prompt (numerical answer) questions and processes accordingly.
    """
    if not isinstance(item_json, dict):
        print(f"Error: preprocess_item expects a dictionary, but received {type(item_json)}. Skipping item from {source_file_path}.")
        return None

    processed_item = {
        "original_id": item_json.get("ID", f"NO_ID_IN_{Path(source_file_path).name}"),
        "source_file": str(source_file_path)
    }

    metadata_fields = ["Assessment", "Test", "Domain", "Skill", "Difficulty"]
    for field in metadata_fields:
        processed_item[field.lower()] = item_json.get(field)

    # --- Text Fields for Parsing and BERT ---
    text_fields_for_parsing = {
        "prompt": item_json.get("Prompt"),
        "paragraph": item_json.get("Paragraph"),
        "stimulus": item_json.get("Stimulus")
    }
    combined_text_for_bert = ""
    for key, text_content in text_fields_for_parsing.items():
        if text_content and isinstance(text_content, str) and text_content.lower() not in ["n/a", "na", ""]:
            cleaned_text = clean_text(text_content)
            processed_item[f"cleaned_{key}"] = cleaned_text
            processed_item[f"parsed_{key}"] = get_syntactic_parse_info(cleaned_text)
            if key in ["prompt", "paragraph", "stimulus"]:
                 combined_text_for_bert += cleaned_text + " "
        else:
            processed_item[f"cleaned_{key}"] = ""
            processed_item[f"parsed_{key}"] = []

    # --- Determine Question Format (Open Prompt vs. Multiple Choice) ---
    correct_answer_raw = item_json.get("Correct Answer")
    is_open_prompt_question = False
    numerical_correct_answer = None

    if correct_answer_raw is not None:
        try:
            if isinstance(correct_answer_raw, (int, float)):
                numerical_correct_answer = correct_answer_raw
                is_open_prompt_question = True
            elif isinstance(correct_answer_raw, str):
                # Avoid misinterpreting 'A', 'B', etc. as part of a number.
                # If it's not a typical single uppercase letter (A-E), try to parse as a number.
                if not (len(correct_answer_raw) == 1 and 'A' <= correct_answer_raw.upper() <= 'E'):
                    numerical_correct_answer = float(correct_answer_raw) # Handles integers, decimals, negatives.
                    is_open_prompt_question = True
        except ValueError:
            # If float conversion fails, and it wasn't an int/float already,
            # assume it's a standard MC label (A, B, C, D).
            is_open_prompt_question = False
        # If it was a single letter A-E, is_open_prompt_question remains False.

    if is_open_prompt_question:
        processed_item["question_format"] = "open_prompt"
        processed_item["correct_answer_value"] = numerical_correct_answer
        processed_item["correct_answer_label"] = None # No A,B,C label for open prompts
        processed_item["correct_answer_text"] = str(numerical_correct_answer)
        processed_item["parsed_correct_answer"] = [] # No meaningful parse for a raw number
        processed_item["answer_choices"] = [] # Ignore any listed answer choices as per requirement
    else:
        # --- Multiple Choice Question Processing ---
        processed_item["question_format"] = "multiple_choice"
        raw_answer_choices = item_json.get("Answer Choices", [])
        processed_item["answer_choices"] = []
        if isinstance(raw_answer_choices, list):
            for i, choice_text in enumerate(raw_answer_choices):
                choice_label = chr(65 + i)
                cleaned_choice = clean_text(str(choice_text))
                processed_item["answer_choices"].append({
                    "label": choice_label,
                    "cleaned_text": cleaned_choice,
                    "parsed_text": get_syntactic_parse_info(cleaned_choice)
                })

        # Correct Answer Label (A, B, C, D) for multiple choice
        mc_correct_label = str(correct_answer_raw).upper() if correct_answer_raw else None
        processed_item["correct_answer_label"] = mc_correct_label
        processed_item["correct_answer_text"] = None # Will be filled if valid label and choices
        processed_item["parsed_correct_answer"] = []


        if mc_correct_label and processed_item["answer_choices"]:
            if 'A' <= mc_correct_label <= 'Z' and len(mc_correct_label) == 1 : # Check for single letter
                correct_index = ord(mc_correct_label) - 65
                if 0 <= correct_index < len(processed_item["answer_choices"]):
                    processed_item["correct_answer_text"] = processed_item["answer_choices"][correct_index]["cleaned_text"]
                    processed_item["parsed_correct_answer"] = processed_item["answer_choices"][correct_index]["parsed_text"]
                else:
                    print(f"Warning: Correct answer label '{mc_correct_label}' out of range for item ID {processed_item['original_id']} in file {source_file_path}")
            else:
                print(f"Warning: Invalid correct answer label format '{mc_correct_label}' for multiple choice item ID {processed_item['original_id']} in file {source_file_path}")
        elif mc_correct_label and not processed_item["answer_choices"]:
             print(f"Warning: Correct answer label '{mc_correct_label}' provided but no answer choices found for item ID {processed_item['original_id']} in file {source_file_path}")


    # --- Rationale (Common to both types) ---
    rationale_text = item_json.get("Rationale")
    if rationale_text and isinstance(rationale_text, str):
        cleaned_rationale = clean_text(rationale_text)
        processed_item["cleaned_rationale"] = cleaned_rationale
        
        rationale_segments = {}
        pattern = r'(Choice\s+([A-Z])\s+(?:is|is not|is the best|is incorrect|is correct|may result|doesn’t|isn’t|presents|doesn\'t|isn\'t)(?:.*?))(?=(?:Choice\s+[A-Z]\s+(?:is|is not|is the best|is incorrect|is correct|may result|doesn’t|isn’t|presents|doesn\'t|isn\'t))|\Z)'
        found_segments = re.finditer(pattern, cleaned_rationale, flags=re.IGNORECASE | re.DOTALL)
        for match in found_segments:
            rationale_segments[match.group(2).upper()] = match.group(1).strip()
        
        if not rationale_segments and "Choice A" in cleaned_rationale and processed_item["question_format"] == "multiple_choice":
            raw_segments = cleaned_rationale.split("Choice ")
            for seg in raw_segments:
                if not seg.strip(): continue
                parts = seg.split(" ", 1)
                if len(parts) > 1:
                    label_candidate = parts[0].replace(".","").replace(":","").strip().upper()
                    if 'A' <= label_candidate <= 'Z' and len(label_candidate) == 1:
                        rationale_segments[label_candidate] = "Choice " + seg.strip()

        processed_item["parsed_rationale_full"] = get_syntactic_parse_info(cleaned_rationale)
        processed_item["segmented_rationale"] = {}
        for label, segment_text_val in rationale_segments.items():
             processed_item["segmented_rationale"][label] = {
                 "text": segment_text_val,
                 "parsed_text": get_syntactic_parse_info(segment_text_val)
             }
    else:
        processed_item["cleaned_rationale"] = ""
        processed_item["parsed_rationale_full"] = []
        processed_item["segmented_rationale"] = {}

    # --- BERT Metadata (Common to both types) ---
    if combined_text_for_bert.strip():
        bert_meta = generate_bert_metadata(combined_text_for_bert.strip())
        processed_item["bert_metadata"] = bert_meta
    else:
        processed_item["bert_metadata"] = {"topics": [], "scores": [], "info": "No sufficient text for BERT analysis."}

    return processed_item


# --- Main Execution ---
if __name__ == "__main__":
    root = Path(__file__).resolve().parents[5]
    data_folder_path_str = root / "question_synthesizer" / "question_data" / "Reading"

    # Example for testing open prompt:
    # data_folder_path_str = r"./test_data" # Create a folder with a sample open prompt JSON
    data_folder = Path(data_folder_path_str)

    all_processed_data = []
    processed_files_count = 0
    processed_items_count = 0
    failed_files_count = 0

    if not data_folder.is_dir():
        print(f"Error: The specified data folder does not exist or is not a directory: {data_folder}")
    else:
        print(f"Starting preprocessing for JSON files in: {data_folder}\n")
        json_files = list(data_folder.rglob("*.json"))

        if not json_files:
            print(f"No JSON files found in {data_folder} or its subdirectories.")
        else:
            print(f"Found {len(json_files)} JSON files to process.")

            for file_path in json_files:
                print(f"Processing file: {file_path}...")
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = json.load(f)
                    
                    current_file_processed_item_count = 0
                    items_to_process = []
                    if isinstance(content, list):
                        items_to_process = content
                        print(f"  File contains a list of {len(content)} items.")
                    elif isinstance(content, dict):
                        items_to_process = [content] # Treat as a list with one item
                        # print("  File contains a single item (dictionary).")
                    else:
                        print(f"  Unsupported JSON structure in {file_path}. Expected dict or list, got {type(content)}.")
                        failed_files_count += 1
                        continue # Skip to next file

                    for item_data in items_to_process:
                        processed = preprocess_item(item_data, source_file_path=file_path)
                        if processed:
                            all_processed_data.append(processed)
                            current_file_processed_item_count += 1
                    
                    if current_file_processed_item_count > 0 :
                        processed_files_count +=1
                        processed_items_count += current_file_processed_item_count
                        print(f"  Successfully processed {current_file_processed_item_count} items from this file.")
                    elif items_to_process : # If there were items but none processed successfully
                        print(f"  Failed to process items from this file.")
                        # Not incrementing failed_files_count here as the file itself might be valid
                        # but items within might be problematic. preprocess_item handles item-level logging.


                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON from file {file_path}: {e}")
                    failed_files_count += 1
                except Exception as e:
                    print(f"An unexpected error occurred while processing file {file_path}: {e}")
                    failed_files_count += 1
                print("-" * 30)

            print(f"\nPreprocessing Summary:")
            print(f"  Total JSON files found: {len(json_files)}")
            print(f"  JSON files from which items were processed: {processed_files_count}")
            print(f"  Total individual question items processed: {processed_items_count}")
            print(f"  Files that failed to load/decode or had unexpected structure: {failed_files_count}")


    if all_processed_data:
        print("\n--- Example Processed Output (First Item Processed) ---")
        first_item_output = all_processed_data[0]
        print(f"Original ID: {first_item_output.get('original_id', 'N/A')}")
        print(f"Source File: {first_item_output.get('source_file', 'N/A')}")
        print(f"Test: {first_item_output.get('test', 'N/A')}")
        print(f"Question Format: {first_item_output.get('question_format', 'N/A')}")
        if first_item_output.get('question_format') == 'open_prompt':
            print(f"Correct Answer Value: {first_item_output.get('correct_answer_value')}")
        else:
            print(f"Correct Answer Label: {first_item_output.get('correct_answer_label')}")
        
        bert_meta = first_item_output.get('bert_metadata', {})
        print(f"BERT Metadata (Top 5 with score > 0.5):")
        if bert_meta.get("topics") and bert_meta.get("scores"):
            threshold = 0.5
            filtered_topics = sorted(
                [(topic, score) for topic, score in zip(bert_meta["topics"], bert_meta["scores"]) if score > threshold],
                key=lambda x: x[1], reverse=True
            )
            if filtered_topics:
                for topic, score in filtered_topics[:5]:
                    print(f"  - Topic: {topic}, Score: {score:.4f}")
            else:
                print("  No topics above threshold or BERT error.")
        elif bert_meta.get("error"):
             print(f"  Error: {bert_meta.get('error')}")
        else:
            print("  No BERT metadata available.")

        output_filename = "question_generator\src\main\backend\rag\preprocess_json_data"
        try:
            with open(output_filename, "w", encoding='utf-8') as f:
                json.dump(all_processed_data, f, indent=4)
            print(f"\nAll {processed_items_count} processed items saved to: {output_filename}")
        except Exception as e:
            print(f"\nError saving processed data to {output_filename}: {e}")
    else:
        print("\nNo data was processed successfully, or no JSON files found.")

    print("\nPreprocessing script finished.")
