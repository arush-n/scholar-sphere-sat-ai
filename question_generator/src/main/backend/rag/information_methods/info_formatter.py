import requests
import xml.etree.ElementTree as ET 
import wikipedia 
import sys
import os

# --- Configuration ---
MAX_RESULTS_PER_SOURCE_RESEARCH = 1 # Max results for research APIs (OpenAlex, arXiv, Semantic Scholar)
REQUEST_TIMEOUT = 10      # Seconds to wait for API responses
WIKIPEDIA_SENTENCES = 0   # 0 for FULL Wikipedia summary.

# --- Path Setup for Local Modules ---
try:
    # IMPORTANT: Replace this with the actual path to your modules if different
    module_base_path = r'C:\Users\rush\OneDrive\Documents\GitHub\scholar-sphere-sat-ai\question_generator\src\main\backend\rag'
    if module_base_path not in sys.path:
        sys.path.insert(0, module_base_path)

    # --- Import Functions from Your Local Modules (using actual names) ---
    from information_methods.wiki import (
        fetch_wikipedia_context,
        fetch_wikidata_context,
        fetch_dbpedia_context
    )
    from information_methods.arxiv import (
        fetch_openalex_context,
        fetch_arxiv_context,
        fetch_semantic_scholar_context,
        reconstruct_abstract_from_inverted_index 
    )
    # Import dictionary function
    from information_methods.dict import fetch_dictionary_definition

    print("[INFO] Successfully imported functions from local modules (wiki, arxiv, dict).")

except ImportError as e:
    print(f"[ERROR] Failed to import local modules: {e}")
    print(f"[ERROR] Please ensure: ")
    print(f"  1. The path '{module_base_path}' is correct.")
    print(f"  2. An empty '__init__.py' file exists in '{os.path.join(module_base_path, 'information_methods')}'.")
    print(f"  3. The files 'knowledge_base_fetchers.py' (or 'wiki.py'), 'research_fetchers.py' (or 'arxiv.py'), and 'dict.py' exist in 'information_methods' and contain the respective functions with the correct names.")
    print("[INFO] Proceeding without local modules if they were not found. Some functionality will be limited.")
    
    # Define dummy functions if imports fail, so the script can partially run
    # These will be overwritten if the specific import succeeded before another failed.
    # This is a simple fallback; more robust handling might be needed for complex dependencies.
    
    # Attempt to define dummy functions only if they haven't been imported.
    # This handles cases where one module might import but another fails.
    if 'fetch_wikipedia_context' not in globals():
        def fetch_wikipedia_context(topic, sentences=0): print(f"[WARN] Wikipedia fetcher not loaded for topic: {topic}"); return None, None
    if 'fetch_wikidata_context' not in globals():
        def fetch_wikidata_context(topic): print(f"[WARN] Wikidata fetcher not loaded for topic: {topic}"); return None
    if 'fetch_dbpedia_context' not in globals():
        def fetch_dbpedia_context(topic): print(f"[WARN] DBpedia fetcher not loaded for topic: {topic}"); return None
    if 'fetch_openalex_context' not in globals():
        def fetch_openalex_context(topic, max_results=1): print(f"[WARN] OpenAlex fetcher not loaded for topic: {topic}"); return []
    if 'fetch_arxiv_context' not in globals():
        def fetch_arxiv_context(topic, max_results=1): print(f"[WARN] arXiv fetcher not loaded for topic: {topic}"); return []
    if 'fetch_semantic_scholar_context' not in globals():
        def fetch_semantic_scholar_context(topic, max_results=1): print(f"[WARN] Semantic Scholar fetcher not loaded for topic: {topic}"); return []
    if 'fetch_dictionary_definition' not in globals():
        def fetch_dictionary_definition(word): 
            print(f"[WARN] Dictionary fetcher not loaded for word: {word}")
            return []

except Exception as e:
    print(f"[ERROR] An unexpected error occurred during module import setup: {e}")
    sys.exit(1)

# --- Wrapper Functions to Format Output for RAG ---

def format_dictionary_api_for_rag(topic):
    """Calls the imported fetch_dictionary_definition and returns its RAG-formatted output."""
    # This wrapper is for consistency in the pipeline.
    # It's particularly useful for single words or short phrases.
    if ' ' not in topic.strip(): # Optional: Only query for single words
        print(f"[INFO] Attempting dictionary definition for single-word topic: '{topic}'")
        # Now calls the (potentially imported) fetch_dictionary_definition
        return fetch_dictionary_definition(topic) 
    else:
        # print(f"[INFO] Skipping dictionary lookup for multi-word topic: '{topic}'") 
        return []


def format_wikipedia_for_rag(topic, sentences=WIKIPEDIA_SENTENCES):
    """Calls original fetch_wikipedia_context and formats for RAG."""
    try:
        # Assumes fetch_wikipedia_context is available (either imported or dummy)
        summary, page_title = fetch_wikipedia_context(topic, sentences=sentences)
        if summary and page_title:
            page_url = ""
            try :
                page = wikipedia.page(page_title, auto_suggest=False, redirect=True) 
                page_url = page.url
            except Exception: 
                page_url = f"https://en.wikipedia.org/wiki/{page_title.replace(' ', '_')}"
            
            return [{
                "source_api": "Wikipedia",
                "source_identifier": page_url,
                "source_title": page_title,
                "context_text": summary
            }]
    except Exception as e:
        print(f"[WARN] Error in format_wikipedia_for_rag for '{topic}': {e}")
    return []

def format_wikidata_for_rag(topic):
    """Calls original fetch_wikidata_context and formats for RAG."""
    try:
        # Assumes fetch_wikidata_context is available
        description = fetch_wikidata_context(topic)
        if description:
            return [{
                "source_api": "Wikidata",
                "source_identifier": f"https://www.wikidata.org/wiki/Special:Search/{topic.replace(' ', '_')}", 
                "source_title": topic, 
                "context_text": description
            }]
    except Exception as e:
        print(f"[WARN] Error in format_wikidata_for_rag for '{topic}': {e}")
    return []

def format_dbpedia_for_rag(topic):
    """Calls original fetch_dbpedia_context and formats for RAG."""
    try:
        # Assumes fetch_dbpedia_context is available
        abstract = fetch_dbpedia_context(topic)
        if abstract:
            key = topic.replace(' ', '_').title() 
            return [{
                "source_api": "DBpedia",
                "source_identifier": f"http://dbpedia.org/resource/{key}",
                "source_title": topic, 
                "context_text": abstract
            }]
    except Exception as e:
        print(f"[WARN] Error in format_dbpedia_for_rag for '{topic}': {e}")
    return []

def format_research_fetcher_for_rag(fetch_function, api_name, topic, max_results=MAX_RESULTS_PER_SOURCE_RESEARCH):
    """Generic wrapper for OpenAlex, arXiv, Semantic Scholar fetchers."""
    rag_contexts = []
    try:
        # Assumes fetch_function (e.g., fetch_openalex_context) is available
        results = fetch_function(topic, max_results=max_results)
        if results: 
            for item in results:
                if not isinstance(item, tuple) or len(item) != 2:
                    print(f"[WARN] Skipping malformed result from {api_name} for topic '{topic}': {item}")
                    continue
                context_text, source_info = item
                
                title = source_info 
                identifier = source_info

                if api_name == "arXiv":
                    arxiv_id_part = source_info.split(':')[-1].strip()
                    identifier = f"http://arxiv.org/abs/{arxiv_id_part}"
                    title = f"arXiv:{arxiv_id_part}" 
                elif api_name == "Semantic Scholar":
                    s2_id_part = source_info.split(':')[-1].strip()
                    identifier = f"https://www.semanticscholar.org/paper/{s2_id_part}"
                    title = f"S2:{s2_id_part}"
                elif api_name == "OpenAlex":
                    if source_info.startswith("https://doi.org/") or source_info.startswith("https://openalex.org/"):
                        identifier = source_info
                    else: 
                        identifier = f"https://openalex.org/works?search={topic.replace(' ', '+')}" 
                    title = f"OpenAlex result for {topic}" 
                
                rag_contexts.append({
                    "source_api": api_name,
                    "source_identifier": identifier,
                    "source_title": title, 
                    "context_text": context_text
                })
        return rag_contexts
    except Exception as e:
        print(f"[WARN] Error in format_research_fetcher_for_rag for '{api_name}' on topic '{topic}': {e}")
    return []

# --- Main Function to Aggregate All Contexts for RAG ---
def get_all_contexts_for_rag(topic):
    """
    Primary function for RAG model. Fetches and formats context from all sources.
    """
    print(f"\n--- Aggregating RAG context for topic: '{topic}' ---")
    aggregated_contexts = []

    # Pipeline using wrapper functions
    fetch_pipeline_wrapped = [
        lambda t: format_dictionary_api_for_rag(t), # Uses imported or dummy dictionary fetcher
        lambda t: format_wikipedia_for_rag(t, sentences=WIKIPEDIA_SENTENCES),
        lambda t: format_wikidata_for_rag(t),
        lambda t: format_dbpedia_for_rag(t),
        lambda t: format_research_fetcher_for_rag(fetch_openalex_context, "OpenAlex", t, MAX_RESULTS_PER_SOURCE_RESEARCH),
        lambda t: format_research_fetcher_for_rag(fetch_arxiv_context, "arXiv", t, MAX_RESULTS_PER_SOURCE_RESEARCH),
        lambda t: format_research_fetcher_for_rag(fetch_semantic_scholar_context, "Semantic Scholar", t, MAX_RESULTS_PER_SOURCE_RESEARCH),
    ]

    for wrapped_func in fetch_pipeline_wrapped:
        try:
            contexts = wrapped_func(topic)
            if contexts: 
                aggregated_contexts.extend(contexts)
        except Exception as e:
            print(f"[ERROR] Unhandled error calling a wrapped fetch function for '{topic}': {e}")

    # --- Enhancement for single-word topics if Wikipedia search yielded no results ---
    is_single_word_topic = ' ' not in topic.strip()
    wikipedia_found_initial_pass = any(item.get('source_api') == 'Wikipedia' for item in aggregated_contexts)

    if is_single_word_topic and not wikipedia_found_initial_pass:
        print(f"[INFO] Single-word topic '{topic}' had no initial Wikipedia hit. Attempting to use Wikidata for a better query.")
        try:
            wikidata_contexts_for_enhancement = format_wikidata_for_rag(topic) 
            if wikidata_contexts_for_enhancement and wikidata_contexts_for_enhancement[0].get('context_text'):
                wikidata_description = wikidata_contexts_for_enhancement[0]['context_text']
                new_wiki_query_parts = wikidata_description.split('.')
                new_wiki_query = new_wiki_query_parts[0].strip() if new_wiki_query_parts else wikidata_description.strip()

                if new_wiki_query and new_wiki_query.lower() != topic.lower():
                    print(f"[INFO] Using new query for Wikipedia based on Wikidata: '{new_wiki_query}'")
                    enhanced_wiki_contexts = format_wikipedia_for_rag(new_wiki_query, sentences=WIKIPEDIA_SENTENCES)
                    if enhanced_wiki_contexts:
                        print(f"[INFO] Found additional Wikipedia context using enhanced query for '{topic}'.")
                        existing_wiki_titles = {ctx.get('source_title') for ctx in aggregated_contexts if ctx.get('source_api') == 'Wikipedia'}
                        for enhanced_ctx in enhanced_wiki_contexts:
                            if enhanced_ctx.get('source_title') not in existing_wiki_titles:
                                aggregated_contexts.append(enhanced_ctx)
                else:
                    print(f"[INFO] Wikidata description for '{topic}' did not yield a substantially different query.")
            else:
                print(f"[INFO] No Wikidata description found for '{topic}' to enhance Wikipedia query.")
        except Exception as e:
            print(f"[WARN] Error during Wikipedia enhancement attempt for '{topic}': {e}")
    
    if not aggregated_contexts:
        print(f"[INFO] No context found for '{topic}' from any source after all attempts.")
    else:
        print(f"\n--- Total RAG-formatted contexts fetched for '{topic}': {len(aggregated_contexts)} ---")
        
    return aggregated_contexts

# --- Example Usage (This block runs only when the script is executed directly) ---
if __name__ == "__main__":
    print("[INFO] Running RAG Context Aggregator in direct execution mode for testing.")

    topics_to_search = [
        "photovoltaic cells", 
        "hello", 
        "ubiquitous", 
        "nonexistentwordxyz" 
    ]

    all_results_for_rag = {}

    for current_topic in topics_to_search:
        rag_data = get_all_contexts_for_rag(current_topic)
        all_results_for_rag[current_topic] = rag_data
        
        if rag_data:
            print(f"\n--- Structured RAG Data for: '{current_topic}' ---")
            for i, item in enumerate(rag_data):
                print(f"\nItem {i+1}:")
                print(f"  Source API: {item.get('source_api', 'N/A')}") 
                print(f"  Source Title: {item.get('source_title', 'N/A')}")
                print(f"  Source ID: {item.get('source_identifier', 'N/A')}")
                context_preview = item.get('context_text', '')
                print(f"  Context (Preview): {context_preview[:250] + '...' if len(context_preview) > 250 else context_preview}")
        else:
            print(f"\n--- No RAG data gathered for '{current_topic}'. ---")
        print("-" * 70)
