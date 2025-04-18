import re
from sympy import sympify
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import io
import numpy as np
import base64

# import graph information

# Mapping for MathJax terms to mathematical symbols
mathjax_to_math = {
    "StartStartFraction": "(",
    "EndEndFraction": ")",
    "OverOver": "/",
    "StartFraction": "(",
    "EndFraction": ")",
    "Over": "/",
    "minus": "-",
    "equals": "=",
    "comma": ",",
    "plus": "+",
    "times": "*",
    "RootIndex": "^1/",
    "left parenthesis": "(",
    "right parenthesis": ")",
    "x 1": "x1",
    "y 1": "y1",
    "x 2": "x2",
    "y 2": "y2",
    "negative": "(-)",
    "greater than or =": ">=",
    "less than or =": "<=",
    "third": "/3",
    "fourth": "/4",
    "half": "/2",
    "halves": "/2",
    "point": ".",
    ".s": "points",   
    "fifth":"/5",
    "sixth":"/6",
    "seventh":"/7",
    "eighth":"/8",
    "ninth":"/9",
    "tenth":"/10",
    "eleventh":"/11",
    "twelfth":"/12",
    "fifteenths":"/15",
    "twentieth":"/20",
    "10th":"/10",
    "Superscript":"^",
    "squared":"^2",
    "cubed":"^3",
    "Baseline":"",
    "dot":"*",
    "one":"1",
    "two":"2",
    "three":"3",
    "four":"4",
    "five":"5",
    "six":"6",
    "seven":"7",
    "eight":"8",
    "nine":"9",
    "ten":"10",
    "eleven":"11",
    "twelve":"12",
    "repeating after the decimal.": "repeating after the decimal",
    "m1y":"money",
    "StartRoot":"sqrt(",
    "EndRoot":")",
    "power": "^",
    "end power": "",
    "denominator": ")/(",
    "numerator": "(",
    "end fraction": ")",
    "subscript": "_",
}

# Function to convert MathJax terms into proper mathematical notation
def reverse_engineer_mathjax(text):
    # First, replace all MathJax terms with the corresponding symbols
    for key, value in mathjax_to_math.items():
        text = text.replace(key, value)

    # Split the text into words and process each one
    words = text.split()
    result = []

    for word in words:
        stripped_word = word.strip()

        if stripped_word == ".":
            # Handle "point" properly for decimal numbers
            if result and result[-1][-1].isdigit():
                result[-1] += "."
            else:
                result.append(".")
        elif stripped_word == ",":
            # Add commas after numbers or variables without extra spaces
            if result and (result[-1][-1].isdigit() or result[-1][-1].isalpha()):
                result[-1] += ","
        elif stripped_word in ["=", "+", "-", "*", "/"]:
            # Ensure operators have spaces around them
            result.append(f" {stripped_word} ")
        else:
            # Ensure there is space between alphabetic characters and numbers
            if result and ((result[-1].isalpha() and stripped_word.isalpha()) or
                           (result[-1].replace('.', '').isdigit() and stripped_word.isalpha()) or
                           (result[-1].isalpha() and stripped_word.replace('.', '').isdigit())):
                result.append(f" {stripped_word}")
            else:
                result.append(stripped_word)

    # Join the processed words into a single string, ensuring proper spaces
    result_text = ' '.join(result)

    # Correct specific phrases to ensure proper spacing and remove misplaced commas
    result_text = (result_text
                   .replace("withthe", "with the")
                   .replace("coordinates", "coordinates ")
                   .replace("thedecimal.", "the decimal.")
                   .replace(",+,", ", +, ")
                   .replace(", + ", "+")
                   .replace(", = ", " = ")
                   .replace(",=", ", = ")
                   .replace("=,", "= ")
                   .replace("*,", "* ")
                   .replace('a,b', '(a, b)')
                   .replace("coordinatesa", "coordinates a")  # Ensure proper spacing for coordinates
                   .replace("coordinatesb", "coordinates b")
                   .replace("withcoordinates", "with coordinates")
                   .replace("the.withcoordinates", "the with coordinates"))

    # Correct parentheses for coordinates, ensuring no double parentheses
    result_text = re.sub(r'coordinates\s*(\d+),\s*(\d+)', r'coordinates (\1, \2)', result_text)
    result_text = re.sub(r'\(\((\d+),\s*(\d+)\)\)', r'(\1, \2)', result_text)  # Fix double parentheses issue

    # Remove any excess multiple spaces and return the final result
    return re.sub(r'\s+', ' ', result_text).strip()


# Function to extract and process MathJax images in the rationale
def extract_rationale_with_mathjax(rationale_element):
    rationale_html = rationale_element.find_next('div')
    rationale_text = ""

    if rationale_html:
        # Iterate through all elements, including MathJax images
        for elem in rationale_html.descendants:
            if elem.name == "img" and "alt" in elem.attrs:
                # If an img tag with alt attribute, process the alt text
                rationale_text += reverse_engineer_mathjax(elem['alt']) + " "
            elif isinstance(elem, str):
                # Add plain text content
                rationale_text += elem.strip() + " "
    
    # Clean up extra spaces
    return rationale_text.strip()

# Function to attempt to simplify or parse mathematical expressions
def parse_math_expression(expression):
    try:
        return str(sympify(expression))
    except:
        return expression

# Function to get the content after a specific label
def get_content_after_label(soup, label):
    element = soup.find('h6', string=label)
    if element:
        sibling = element.find_next_sibling('div')
        return sibling.text.strip() if sibling else None
    return None

# Replace MathJax elements with alttext, and then reverse engineer the alttext into math symbols
def replace_mathjax_and_reverse_engineer(soup):
    for mjx_container in soup.find_all('mjx-container', {'alttext': True}):
        alt_text = mjx_container['alttext']
        math_expression = reverse_engineer_mathjax(alt_text)
        simplified_expression = parse_math_expression(math_expression)
        mjx_container.replace_with(simplified_expression)

# Function to visualize the table using matplotlib and return as base64-encoded image
def visualize_table(headers, rows, title):
    num_columns = len(headers)
    num_rows = len(rows)

    # Adjust figure size dynamically based on the number of rows and columns
    fig_width = max(6, num_columns * 1.2)  # Minimum width of 6, scaled by number of columns
    fig_height = max(3.5, num_rows * 0.6)  # Minimum height of 3.5, scaled by number of rows
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))  # Set dynamic size based on content
    ax.axis('tight')
    ax.axis('off')

    # Create the table with centered text
    table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center', colColours=["#f8f8f8"] * num_columns)

    # Set table title
    plt.title(title, fontsize=12, weight='bold')

    # Set uniform column width and row height for all cells
    col_width = 1 / num_columns  # Set uniform column width as a fraction of total
    row_height = 1 / (num_rows + 1)  # Set uniform row height, including header

    # Set the font size for all cells
    table.auto_set_font_size(False)
    table.set_fontsize(12)  # Adjust font size to fit text

    # Adjust table layout by setting equal width for all columns and height for all rows
    for i in range(num_rows + 1):  # Including header row
        for j in range(min(num_columns, len(headers))):  # Avoid out-of-bound errors
            try:
                cell = table[(i, j)]
                cell.set_width(col_width)  # Set uniform column width
                cell.set_height(row_height)  # Set uniform row height
            except KeyError:
                print(f"Skipping cell ({i}, {j}) due to KeyError")

    # Save the figure to a BytesIO object and encode it in base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    plt.close()  # Close the plot to free up memory
    buffer.seek(0)

    # Encode the image in base64 and return it
    return base64.b64encode(buffer.getvalue()).decode()

# Function to convert textual coordinates to numeric coordinates
def text_to_coordinates(text):
    text = text.replace('negative', '-').replace('comma', ',').replace(' ', '')
    points = re.findall(r'\((-?\d*\.?\d*)\s*,\s*(-?\d*\.?\d*)\)', text)

    try:
        coordinates = [tuple(map(float, point)) for point in points]
    except ValueError as e:
        print(f"Error parsing coordinates: {e}")
        coordinates = []
    return coordinates

# Extract table content, title, and visualize using matplotlib
def extract_and_visualize_table(soup):
    table = soup.find('table')
    if table:
        # Get table title if available
        title_tag = table.find('caption')
        title = title_tag.text.strip() if title_tag else 'Table'

        # Extract headers
        headers = [th.text.strip() for th in table.find_all('th')]

        # Extract rows, including the first column with street names
        rows = []
        for tr in table.find_all('tr')[1:]:  # Skip header row
            row_data = [td.text.strip() for td in tr.find_all(['td', 'th'])]  # Include 'th' for row headers (e.g., street names)
            rows.append(row_data)

        return visualize_table(headers, rows, title)
    return None

# Extract stem paragraph after the table
def extract_stem_paragraph(soup):
    # Find the table first and then get the paragraph after the table
    table = soup.find('table')
    if table:
        stem_paragraph = table.find_next('p')
        if stem_paragraph:
            return stem_paragraph.text.strip()
    return None

# Function to format the prompt and handle line breaks and proper spacing
def format_prompt(text):
    print('text', text)
    # Replace MathJax terms using the dictionary
    for key, value in mathjax_to_math.items():
        text = text.replace(key, value)

    text = text.replace("<br/>", " ")  # Replace line breaks with space for now
    # Ensure proper spacing around numbers, variables (x, y), and operators (=, +, -)
    text = re.sub(r'([0-9xy])([=+\-])', r'\1 \2 ', text)  # Space after numbers/vars and operators
    text = re.sub(r'([=+\-])([0-9xy])', r' \1 \2', text)  # Space before numbers/vars and operators
    
    # Ensure there's space between sentences and before/after line breaks
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space

    return text.strip()

def q_parse_no_images(html, genre):
    html_content = html

    # Parse the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')

    # Handle the prompt properly, including <br> tags and other HTML structures
    prompt_div = soup.find('div', class_='question')

    # Main extraction and processing logic
    replace_mathjax_and_reverse_engineer(soup)

    # Extract assessment, test, domain, skill, and difficulty
    question_info = {}
    question_info['Assessment'] = get_content_after_label(soup, 'Assessment')
    question_info['Test'] = get_content_after_label(soup, 'Test')
    question_info['Domain'] = get_content_after_label(soup, 'Domain')
    question_info['Skill'] = get_content_after_label(soup, 'Skill')

    # Extract difficulty (handle differently since it's in a span with 'aria-label')
    difficulty_element = soup.find('span', {'aria-label': True})
    question_info['Difficulty'] = difficulty_element['aria-label'] if difficulty_element else None

    # Extract the question ID and prompt
    question_info['ID'] = soup.find('h5', class_='question-id').text.replace('ID: ', '').strip()

    # Handle the prompt: first try to extract image in background if present, otherwise fall back to text
    prompt_div = soup.find('div', class_='question')

    if prompt_div:
        # Replace all <br> tags with a newline character (\n)
        for br in prompt_div.find_all("br"):
            br.replace_with("\n")


        full_prompt = prompt_div.text.strip()
        formatted_prompt = format_prompt(full_prompt)  # Apply the formatting for proper spacing
        question_info['Prompt'] = formatted_prompt

    else:
        question_info['Prompt'] = "Prompt not found"
        
    if genre.lower() == 'reading':
        paragraph_content = ""
        paragraph_tags = soup.find('p')
        for paragraph in paragraph_tags:
            paragraph_content += paragraph.get_text(strip=True) + "\n"
            
        question_info['Paragraph'] = paragraph_content.strip() if paragraph_content else "No paragraph found"
        # Find an <img> element with an aria-label
        image_element = soup.find('img', {'aria-label': True})
        table_element = soup.find('table')
                
        if image_element:
            # Return the aria-label if it's present
            question_info['Stimulus'] = image_element['aria-label']
        else:
            # Otherwise return unreadable image
            question_info['Stimulus'] = "question contains unreadable image"
        
        if table_element:
            # Extract all rows from the table
            rows = table_element.find_all('tr')
            
            table_data = []
            for row in rows:
                # Get all columns (th or td) from the row
                columns = row.find_all(['th', 'td'])
                # Extract text and join with commas
                row_data = ', '.join(col.get_text(strip=True) for col in columns)
                table_data.append(row_data)
            
            # Join all rows with new lines
            question_info['Stimulus'] = '\n'.join(table_data)
        else:
            question_info['Stimulus'] = "N/A"

        
    else:
        # Check if the prompt contains an image
        if prompt_div.find('img'):
            return 'question contains image'
        
    # Extract answer choices (look for <li> elements) and filter to keep only the final four
    answer_choices = []
    all_li_tags = soup.find_all('li')

    for li in all_li_tags[-4:]:
        img_tag = li.find('img')
        if img_tag:
            return 'question contains image'
        else:
            text_in_li = li.get_text(strip=True)
            clean_text = reverse_engineer_mathjax(text_in_li)
            answer_choices.append(clean_text)

    # Store the answer choices in question_info
    question_info['Answer Choices'] = answer_choices if answer_choices else 'N/A'

    # Extract the correct answer and rationale
    correct_answer_element = soup.find('h6', string='Correct Answer: ')
    question_info['Correct Answer'] = correct_answer_element.find_next('p').text.strip() if correct_answer_element else None

    rationale_element = soup.find('h6', string='Rationale')
    if rationale_element:
        question_info['Rationale'] = extract_rationale_with_mathjax(rationale_element)
    else:
        question_info['Rationale'] = None

    # Print the extracted and corrected information
    for key, value in question_info.items():
        if key in ["Table Image", "Graph"]:
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value}")

    return question_info

# TODO - 1. MATH EXPRESSION - MATH CONTAINER DETECTION (USE IMAGE) 2. IDENTIFYING LONG DESCRIPTIONS 3. ENSURE IMAGES ALT TEXT FOR ANSWER CHOICES IS ALSO THERE FOR TRAINING PURPOSES
def q_parse(html):
    # Open and read the HTML file
    with open(html, "r", encoding="utf-8") as file:
        html_content = file.read()

    # Parse the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')

    # Handle the prompt properly, including <br> tags and other HTML structures
    prompt_div = soup.find('div', class_='question')

    # Main extraction and processing logic
    replace_mathjax_and_reverse_engineer(soup)

    # Extract assessment, test, domain, skill, and difficulty
    question_info = {}
    question_info['Assessment'] = get_content_after_label(soup, 'Assessment')
    question_info['Test'] = get_content_after_label(soup, 'Test')
    question_info['Domain'] = get_content_after_label(soup, 'Domain')
    question_info['Skill'] = get_content_after_label(soup, 'Skill')

    # Extract difficulty (handle differently since it's in a span with 'aria-label')
    difficulty_element = soup.find('span', {'aria-label': True})
    question_info['Difficulty'] = difficulty_element['aria-label'] if difficulty_element else None

    # Extract the question ID and prompt
    question_info['ID'] = soup.find('h5', class_='question-id').text.replace('ID: ', '').strip()

    # Handle the prompt: first try to extract image in background if present, otherwise fall back to text
    prompt_div = soup.find('div', class_='question')

    if prompt_div:
        # Replace all <br> tags with a newline character (\n)
        for br in prompt_div.find_all("br"):
            br.replace_with("\n")

        full_prompt = prompt_div.text.strip()
        formatted_prompt = format_prompt(full_prompt)  # Apply the formatting for proper spacing
        question_info['Prompt'] = formatted_prompt

    else:
        question_info['Prompt'] = "Prompt not found"

    # Handle the case where a table is present
    if soup.find('table'):
        table_image_base64 = extract_and_visualize_table(soup)
        if table_image_base64:
            question_info['Table Image'] = f"data:image/png;base64,{table_image_base64}"
        stem_paragraph = extract_stem_paragraph(soup)
        if stem_paragraph:
            question_info['Stem Paragraph'] = stem_paragraph

    # Extract answer choices (look for <li> elements) and filter to keep only the final four
    answer_choices = []
    all_li_tags = soup.find_all('li')

    for li in all_li_tags[-4:]:
        img_tag = li.find('img')
        if img_tag:
            img_src = img_tag.get('src')
            if img_src:
                answer_choices.append(f"[IMAGE: {img_src}]")
        else:
            text_in_li = li.get_text(strip=True)
            clean_text = reverse_engineer_mathjax(text_in_li)
            answer_choices.append(clean_text)

    # Store the answer choices in question_info
    question_info['Answer Choices'] = answer_choices if answer_choices else 'N/A'

    # Extract the correct answer and rationale
    correct_answer_element = soup.find('h6', string='Correct Answer: ')
    question_info['Correct Answer'] = correct_answer_element.find_next('p').text.strip() if correct_answer_element else None

    rationale_element = soup.find('h6', string='Rationale')
    if rationale_element:
        question_info['Rationale'] = extract_rationale_with_mathjax(rationale_element)
    else:
        question_info['Rationale'] = None

    # Print the extracted and corrected information
    for key, value in question_info.items():
        if key in ["Table Image", "Graph"]:
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value}")

# TODO - 1. MATH EXPRESSION - MATH CONTAINER DETECTION (USE IMAGE) 2. IDENTIFYING LONG DESCRIPTIONS 3. ENSURE IMAGES ALT TEXT FOR ANSWER CHOICES IS ALSO THERE FOR TRAINING PURPOSES