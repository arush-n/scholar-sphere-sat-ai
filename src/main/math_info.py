import no_images_webscraper as ws

genre  = 'Math'
math_domain = {'Algebra': '//*[@id="checkbox-algebra"]', 'Advanced Math': '//*[@id="checkbox-advanced math"]', 'Problem-Solving and Data Analysis': '//*[@id="checkbox-problem-solving and data analysis"]','Geometry and Trigonometry': '//*[@id="checkbox-geometry and trigonometry"]'}
math_topics = {
    'Algebra': {
        'Linear equations in one variable': '//*[@id="Linear equations in one variable"]',
        'Linear functions': '//*[@id="Linear functions"]',
        'Linear equations in two variables': '//*[@id="Linear equations in two variables"]',
        'Systems of two linear equations in two variables': '//*[@id="Systems of two linear equations in two variables"]',
        'Linear inequalities in one or two variables': '//*[@id="Linear inequalities in one or two variables"]'
    },
    'Advanced Math': {
        'Nonlinear functions': '//*[@id="Nonlinear functions"]',
        'Nonlinear equations in one variable and systems of equations in two variables': '//*[@id="Nonlinear equations in one variable and systems of equations in two variables"]',
        'Equivalent expressions': '//*[@id="Equivalent expressions"]'
    },
        'Problem-Solving and Data Analysis': {
        'Ratios, rates, proportional relationships, and units': '//*[@id="Ratios, rates, proportional relationships, and units"]',
        'Percentages': '//*[@id="Percentages"]',
        'One-variable data: Distributions and measures of center and spread': '//*[@id="One-variable data: Distributions and measures of center and spread"]',
        'Two-variable data: Models and scatterplots': '//*[@id="Two-variable data: Models and scatterplots"]',
        'Probability and conditional probability': '//*[@id="Probability and conditional probability"]',
        'Inference from sample statistics and margin of error': '//*[@id="Inference from sample statistics and margin of error"]',
        'Evaluating statistical claims: Observational studies and experiments': '//*[@id="Evaluating statistical claims: Observational studies and experiments"]'
    },
    'Geometry and Trigonometry': {
        'Area and volume': '//*[@id="Area and volume"]',
        'Lines angles and triangles': '//*[@id="Lines, angles, and triangles"]',
        'Right triangles and trigonometry': '//*[@id="Right triangles and trigonometry"]',
        'Circles': '//*[@id="Circles"]'
    }
}
exclude_active = False

# Loop through each domain and topic
for domain, topics in math_topics.items():
    for topic, xpath in topics.items():
        print(f"Scraping for domain: {domain}, topic: {topic}, xpath: {xpath}")
        # Call the webscraper function, passing the necessary parameters
        ws.webscraper(genre, {domain: math_domain[domain]}, xpath, exclude_active)

