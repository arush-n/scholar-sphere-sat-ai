import requests
import wikipedia

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