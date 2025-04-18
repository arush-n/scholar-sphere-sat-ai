import no_images_webscraper as ws

genre  = 'Reading'
english_domain = {'Information and Ideas': '//*[@id="checkbox-information and ideas"]', 'Craft and Structure': '//*[@id="checkbox-craft and structure"]', 'Expression of Ideas': '//*[@id="checkbox-expression of ideas"]', 'Standard English Conventions': '//*[@id="checkbox-standard english conventions"]'}
english_topics = {
    'Information and Ideas': {
        'Central Ideas and Details': '//*[@id="Central Ideas and Details"]',
        'Inferences': '//*[@id="Inferences"]',
        'Command of Evidence': '//*[@id="Command of Evidence"]'
    },
    'Craft and Structure': {
        'Words in Context': '//*[@id="Words in Context"]',
        'Text Structure and Purpose': '//*[@id="Text Structure and Purpose"]',
        'Cross-Text Connections': '//*[@id="Cross-Text Connections"]'
    },
    'Expression of Ideas': {
        'Rhetorical Synthesis': '//*[@id="Rhetorical Synthesis"]',
        'Transitions': '//*[@id="Transitions"]'
    },
    'Standard English Conventions': {
        'Boundaries': '//*[@id="Boundaries"]',
        'Form, Structure, and Sense': '//*[@id="Form, Structure, and Sense"]'
    }
}

exclude_active = False

# Loop through each domain and topic
for domain, topics in english_topics.items():
    for topic, xpath in topics.items():
        print(f"Scraping for domain: {domain}, topic: {topic}, xpath: {xpath}")
        # Call the webscraper function, passing the necessary parameters
        ws.webscraper(genre, {domain: english_domain[domain]}, xpath, exclude_active)

