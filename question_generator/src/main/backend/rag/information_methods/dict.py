import requests
import os

# Configuration for the dictionary API
REQUEST_TIMEOUT = 10  # Seconds to wait for API responses

def fetch_dictionary_definition(word):
    """
    Fetches dictionary definitions for a given word using dictionaryapi.dev.

    Args:
        word (str): The word to define.

    Returns:
        list: A list of dictionaries, where each dictionary contains RAG-formatted context
              (source_api, source_identifier, source_title, context_text).
              Returns an empty list if no definition is found or an error occurs.
    """
    api_url = f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}"
    rag_contexts = []
    
    print(f"[INFO] Fetching dictionary definition for '{word}' from {api_url}")

    try:
        response = requests.get(api_url, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()  # Raise an exception for HTTP errors (4xx or 5xx)
        
        data = response.json()

        if isinstance(data, list) and data:
            # Process the first entry returned by the API
            entry = data[0]
            word_returned = entry.get("word", word) # API might return a slightly different form

            # Concatenate definitions from all meanings
            all_definitions_text = []

            for meaning in entry.get("meanings", []):
                part_of_speech = meaning.get("partOfSpeech", "N/A")
                definitions_for_pos = []
                for definition_obj in meaning.get("definitions", []):
                    definition_text = definition_obj.get("definition")
                    if definition_text:
                        definitions_for_pos.append(definition_text)
                
                if definitions_for_pos:
                    all_definitions_text.append(f"As {part_of_speech}: {'; '.join(definitions_for_pos)}")
            
            if all_definitions_text:
                full_context = "\n".join(all_definitions_text)
                rag_contexts.append({
                    "source_api": "DictionaryAPI",
                    "source_identifier": api_url,
                    "source_title": f"Definition of '{word_returned}'",
                    "context_text": full_context.strip()
                })
                print(f"[LOG] Fetched dictionary definition for '{word_returned}'.")
            else:
                print(f"[INFO] No definitions found within the API response structure for '{word}'.")
        else:
            print(f"[WARN] No dictionary definition found or unexpected format for '{word}'. API Response: {data}")
            
    except requests.exceptions.HTTPError as http_err:
        if response.status_code == 404:
            print(f"[INFO] Dictionary definition not found for '{word}' (404 Error).")
        else:
            print(f"[ERROR] HTTP error fetching dictionary definition for '{word}': {http_err}")
    except requests.exceptions.RequestException as req_err:
        print(f"[ERROR] Request error fetching dictionary definition for '{word}': {req_err}")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred fetching dictionary definition for '{word}': {e}")
        
    return rag_contexts

if __name__ == "__main__":
    # Example Usage
    test_words = ["hello", "programming", "ubiquitous", "nonexistentwordxyz"]
    for test_word in test_words:
        print(f"\n--- Testing word: {test_word} ---")
        definitions = fetch_dictionary_definition(test_word)
        if definitions:
            for i, definition_entry in enumerate(definitions):
                print(f"  Source API: {definition_entry['source_api']}")
                print(f"  Source Title: {definition_entry['source_title']}")
                print(f"  Source ID: {definition_entry['source_identifier']}")
                print(f"  Context: {definition_entry['context_text'][:200]}...") # Preview
        else:
            print(f"  No definition data retrieved for '{test_word}'.")
        print("-" * 20)
