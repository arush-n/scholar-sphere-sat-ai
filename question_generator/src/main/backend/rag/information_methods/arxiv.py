import requests
import xml.etree.ElementTree as ET # For parsing arXiv XML

# --- Configuration ---
# You can adjust these global settings
MAX_RESULTS_PER_SOURCE = 1 # Max results to fetch from each source
REQUEST_TIMEOUT = 10       # Seconds to wait for API responses

# --- Helper Function for OpenAlex Abstract Reconstruction ---
def reconstruct_abstract_from_inverted_index(inverted_index):
    """
    Reconstructs an abstract string from OpenAlex's inverted index format.
    Example: {"Abstract": [0], "text": [1]} -> "Abstract text"
    """
    if not inverted_index:
        return ""
    
    # Determine the length of the abstract
    # The highest index value indicates the number of words - 1
    length = 0
    for word_indices in inverted_index.values():
        for index in word_indices:
            if index > length:
                length = index
    length += 1 # To get the actual number of words

    # Initialize a list of Nones with the determined length
    abstract_list = [None] * length
    
    # Populate the list with words based on their indices
    for word, indices in inverted_index.items():
        for index in indices:
            if 0 <= index < length: # Basic bounds check
                abstract_list[index] = word
            else:
                # This case should ideally not happen if the inverted_index is well-formed
                print(f"[WARN] Word '{word}' has out-of-bounds index {index} for abstract length {length}")


    # Join the words, filtering out any None values (if any words were missing)
    return " ".join(filter(None, abstract_list))

# --- API Fetching Functions ---

def fetch_openalex_context(topic, max_results=MAX_RESULTS_PER_SOURCE):
    """
    Fetches context from OpenAlex.
    OpenAlex provides data on scholarly works, authors, institutions, etc.
    No API key needed for the "polite" tier (1 req/sec).
    """
    base_url = "https://api.openalex.org/works"
    params = {
        'search': topic,
        'per_page': max_results,
        # 'select': 'id,doi,title,publication_year,abstract_inverted_index,primary_location' 
        # Select specific fields to reduce response size if needed
    }
    headers = {'Accept': 'application/json'}
    contexts = []

    try:
        response = requests.get(base_url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()

        if data.get('results'):
            for i, work in enumerate(data['results'][:max_results]):
                title = work.get('title', 'N/A')
                # OpenAlex provides abstracts in an "inverted index" format.
                # We need to reconstruct it. Some entries might have a direct 'abstract' field,
                # but it's often null. abstract_inverted_index is more common if abstract exists.
                abstract_text = ""
                if work.get('abstract_inverted_index'):
                    abstract_text = reconstruct_abstract_from_inverted_index(work['abstract_inverted_index'])
                elif work.get('abstract'): # Fallback though less common
                     abstract_text = work.get('abstract', "")

                if abstract_text:
                    source_id = work.get('doi') or work.get('id') or f"OpenAlex: {title}"
                    print(f"[LOG] Fetched OpenAlex context for '{title}'.")
                    contexts.append((abstract_text.strip(), source_id))
                else:
                    print(f"[INFO] No abstract found for OpenAlex work: '{title}'.")
            if contexts:
                return contexts # Returns a list of (context, source_id) tuples
        print(f"[WARN] No OpenAlex results for '{topic}'.")
        return []
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] OpenAlex request failed for '{topic}': {e}")
        return []
    except Exception as e:
        print(f"[ERROR] Error processing OpenAlex data for '{topic}': {e}")
        return []


def fetch_arxiv_context(topic, max_results=MAX_RESULTS_PER_SOURCE):
    """
    Fetches summaries from arXiv, a repository for scientific preprints.
    No API key required for basic use.
    """
    base_url = "http://export.arxiv.org/api/query"
    # arXiv API prefers specific field queries, e.g., ti:topic for title, abs:topic for abstract
    # Using 'all:' for a general search.
    params = {
        'search_query': f'all:{topic}',
        'start': 0,
        'max_results': max_results,
        'sortBy': 'relevance' # Other options: 'lastUpdatedDate', 'submittedDate'
    }
    contexts = []

    try:
        response = requests.get(base_url, params=params, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        
        # Parse XML response
        root = ET.fromstring(response.content)
        atom_ns = '{http://www.w3.org/2005/Atom}' # Atom namespace

        entries = root.findall(f'{atom_ns}entry')
        if not entries:
            print(f"[WARN] No arXiv results for '{topic}'.")
            return []

        for entry in entries:
            title_element = entry.find(f'{atom_ns}title')
            summary_element = entry.find(f'{atom_ns}summary')
            id_element = entry.find(f'{atom_ns}id') # This is a URL like http://arxiv.org/abs/2001.00001v1

            title = title_element.text.strip() if title_element is not None else "N/A"
            summary = summary_element.text.strip() if summary_element is not None else ""
            # Extract the arXiv ID from the URL
            arxiv_id = id_element.text.split('/abs/')[-1] if id_element is not None and id_element.text else title
            
            if summary:
                print(f"[LOG] Fetched arXiv context for '{title}' (ID: {arxiv_id}).")
                contexts.append((summary, f"arXiv:{arxiv_id}"))
            else:
                print(f"[INFO] No summary found for arXiv entry: '{title}'.")
        
        return contexts # Returns a list of (context, source_id) tuples

    except requests.exceptions.RequestException as e:
        print(f"[ERROR] arXiv request failed for '{topic}': {e}")
        return []
    except ET.ParseError as e:
        print(f"[ERROR] Failed to parse arXiv XML response for '{topic}': {e}")
        return []
    except Exception as e:
        print(f"[ERROR] Error processing arXiv data for '{topic}': {e}")
        return []


def fetch_semantic_scholar_context(topic, max_results=MAX_RESULTS_PER_SOURCE):
    """
    Fetches abstracts from Semantic Scholar.
    No API key required for basic use, but rate limits apply (100 reqs/5 mins unauthenticated).
    """
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        'query': topic,
        'limit': max_results,
        'fields': 'title,abstract,paperId,year,authors' # Specify fields to retrieve
    }
    headers = {'Accept': 'application/json'}
    contexts = []

    try:
        response = requests.get(base_url, params=params, headers=headers, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()

        if data.get('data'):
            for paper in data['data']:
                title = paper.get('title', 'N/A')
                abstract = paper.get('abstract', '')
                paper_id = paper.get('paperId', title) # Use paperId or fallback to title

                if abstract:
                    print(f"[LOG] Fetched Semantic Scholar context for '{title}'.")
                    contexts.append((abstract.strip(), f"S2:{paper_id}"))
                else:
                    print(f"[INFO] No abstract found for Semantic Scholar paper: '{title}'.")
            if contexts:
                return contexts # Returns a list of (context, source_id) tuples

        # Handle cases where 'data' might be empty or not present
        total_results = data.get('total', 0)
        if total_results == 0:
            print(f"[WARN] No Semantic Scholar results for '{topic}'.")
        elif not contexts: # Results found but no abstracts perhaps
             print(f"[WARN] Semantic Scholar results found for '{topic}', but no usable abstracts retrieved.")
        return []
        
    except requests.exceptions.RequestException as e:
        print(f"[ERROR] Semantic Scholar request failed for '{topic}': {e}")
        return []
    except Exception as e:
        print(f"[ERROR] Error processing Semantic Scholar data for '{topic}': {e}")
        return []

# --- Main Function to Aggregate Context ---
def get_combined_research_context(topic, num_sentences_per_source=None):
    """
    Fetches context from multiple research sources for a given topic.
    `num_sentences_per_source` is not directly implemented here as APIs return abstracts/summaries.
    You might add post-processing to truncate to N sentences if needed.
    """
    print(f"\n--- Fetching context for topic: '{topic}' ---")
    
    all_contexts = []

    # Fetch from OpenAlex
    openalex_contexts = fetch_openalex_context(topic)
    if openalex_contexts:
        all_contexts.extend(openalex_contexts)

    # Fetch from arXiv
    arxiv_contexts = fetch_arxiv_context(topic)
    if arxiv_contexts:
        all_contexts.extend(arxiv_contexts)

    # Fetch from Semantic Scholar
    semantic_scholar_contexts = fetch_semantic_scholar_context(topic)
    if semantic_scholar_contexts:
        all_contexts.extend(semantic_scholar_contexts)
    
    if not all_contexts:
        print(f"[INFO] No context found for '{topic}' from any source.")
        return [] # Return empty list if no context found

    print(f"\n--- Total contexts fetched for '{topic}': {len(all_contexts)} ---")
    return all_contexts # Returns a list of (context_string, source_identifier) tuples

# --- Example Usage ---
if __name__ == "__main__":
    # Example topics to test
    topics = [
        "artificial intelligence in healthcare",
        "quantum entanglement applications",
        "crispr gene editing ethics",
        "dark matter detection methods",
        "nonexistent topic xyz123abc" # To test no results
    ]

    for t in topics:
        contexts_found = get_combined_research_context(t)
        if contexts_found:
            for i, (context, source_id) in enumerate(contexts_found):
                print(f"\nContext {i+1} from {source_id}:")
                # Print first 300 chars for brevity in example output
                print(context[:300] + "..." if len(context) > 300 else context)
        else:
            print(f"No research context gathered for '{t}'.")
        print("-" * 50)

    # Example of fetching more results from one source
    print("\n--- Fetching more results for a single topic from arXiv ---")
    single_topic = "machine learning interpretability"
    arxiv_more_results = fetch_arxiv_context(single_topic, max_results=3)
    if arxiv_more_results:
        for i, (context, source_id) in enumerate(arxiv_more_results):
            print(f"\nContext {i+1} from {source_id}:")
            print(context[:300] + "..." if len(context) > 300 else context)
    else:
        print(f"No arXiv context found for '{single_topic}' with more results.")

    print("\n--- Fetching more results for a single topic from OpenAlex ---")
    single_topic_oa = "climate change impact on biodiversity"
    oa_more_results = fetch_openalex_context(single_topic_oa, max_results=2)
    if oa_more_results:
        for i, (context, source_id) in enumerate(oa_more_results):
            print(f"\nContext {i+1} from {source_id}:")
            print(context[:300] + "..." if len(context) > 300 else context)
    else:
        print(f"No OpenAlex context found for '{single_topic_oa}' with more results.")

