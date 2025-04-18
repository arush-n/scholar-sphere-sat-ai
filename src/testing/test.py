import re
from sympy import sympify
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import io
import numpy as np
import base64

#from src.graphs import line_chart as lc


# Open and read the HTML file
with open("src\\testing\\test.html", "r", encoding="utf-8") as file:
    html_content = file.read()

# Parse the HTML content
soup = BeautifulSoup(html_content, 'html.parser')

# Mapping for MathJax terms to mathematical symbols
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

def generate_two_linear_equation_graph(prompt):
    if 'For the first line in the system' in prompt and 'For the second line in the system' in prompt:
        # Extract points for the first line
        first_line_text = prompt.split('For the first line in the system:')[1].split('For the second line in the system:')[0]
        first_line_points = text_to_coordinates(first_line_text)
        x1, y1 = zip(*first_line_points) if first_line_points else ([], [])

        # Extract points for the second line
        second_line_text = prompt.split('For the second line in the system:')[1]
        second_line_points = text_to_coordinates(second_line_text)
        x2, y2 = zip(*second_line_points) if second_line_points else ([], [])

        return generate_base64_graph(x1, y1, x2, y2)
    return None

def generate_single_linear_equation_graph(prompt):
    # Check if the prompt contains the phrase indicating a single line equation
    if 'The line passes through the following points:' in prompt:
        try:
            # Extract points after 'The line passes through the following points:'
            line_text = prompt.split('The line passes through the following points:')[1]
            line_points = text_to_coordinates(line_text)
            x, y = zip(*line_points) if line_points else ([], [])

            return generate_base64_graph_single_line(x, y)
        except IndexError:
            print("Error: Unable to parse line points from the prompt.")
            return None
    return None

# Function to generate a graph for a single linear equation and return it as a base64-encoded string
def generate_base64_graph_single_line(x, y):
    # Determine the dynamic size based on the range of x and y values
    x_min, x_max = min(min(x), -10), max(max(x), 10)
    y_min, y_max = min(min(y), -10), max(max(y), 10)

    # Extend the axis limits slightly beyond the data points
    x_margin = (x_max - x_min) * 0.2  # 20% margin
    y_margin = (y_max - y_min) * 0.2  # 20% margin
    
    x_min, x_max = x_min - x_margin, x_max + x_margin
    y_min, y_max = y_min - y_margin, y_max + y_margin
    
    # Ensure the aspect ratio is 1:1 by adjusting the range of x and y equally
    max_range = max(x_max - x_min, y_max - y_min)
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    x_min, x_max = x_center - max_range / 2, x_center + max_range / 2
    y_min, y_max = y_center - max_range / 2, y_center + max_range / 2

    # Set the figure size proportional to the ranges while maintaining the aspect ratio
    plt.figure(figsize=(max(5, max_range / 2), max(5, max_range / 2)))

    # Use multiples of 2 for grid intervals
    x_interval = y_interval = 2

    # Plot the line with circles for points
    if x and y:
        slope = (y[-1] - y[0]) / (x[-1] - x[0]) if (x[-1] - x[0]) != 0 else np.inf
        intercept = y[0] - slope * x[0]
        x_vals = np.array([x_min, x_max])  # Extend line to the plot limits
        plt.plot(x_vals, slope * x_vals + intercept, color='black')

        # Plot circles at the points
        plt.scatter(x, y, color='black', s=100, zorder=5, edgecolor='black', label='Points on line')

    # Set the aspect ratio of the plot to be square (1:1 aspect ratio)
    plt.gca().set_aspect('equal', adjustable='box')

    # Set the x and y axis limits
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # Add gridlines with intervals of 2
    plt.grid(True, which='both', linestyle='--', linewidth=0.7)
    
    # Set ticks based on intervals of 2
    plt.xticks(np.arange(x_min, x_max, x_interval))
    plt.yticks(np.arange(y_min, y_max, y_interval))

    # Show the x and y axes
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)

    # Remove the external tick labels but keep gridlines
    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])

    # Add labels for axes directly on the x and y axes without decimal points
    for i in np.arange(x_min, x_max, x_interval):
        if i != 0:
            plt.text(i, 0.5, f'{int(i)}', ha='center', va='center', fontsize=10, fontweight='bold')

    for i in np.arange(y_min, y_max, y_interval):
        if i != 0:
            plt.text(0.5, i, f'{int(i)}', ha='center', va='center', fontsize=10, fontweight='bold')

    # Save the plot to a BytesIO object and encode it in base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    plt.close()
    buffer.seek(0)

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

# Function to generate a graph and return it as a base64-encoded string
def generate_base64_graph(x1, y1, x2, y2):
    # Determine the dynamic size based on the range of x and y values
    x_min, x_max = min(min(x1 + x2), -10), max(max(x1 + x2), 10)
    y_min, y_max = min(min(y1 + y2), -10), max(max(y1 + y2), 10)
    
    # Extend the axis limits slightly beyond the data points
    x_margin = (x_max - x_min) * 0.2  # 20% margin
    y_margin = (y_max - y_min) * 0.2  # 20% margin
    
    x_min, x_max = x_min - x_margin, x_max + x_margin
    y_min, y_max = y_min - y_margin, y_max + y_margin
    
    # Ensure the aspect ratio is 1:1 by adjusting the range of x and y equally
    max_range = max(x_max - x_min, y_max - y_min)
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2
    x_min, x_max = x_center - max_range / 2, x_center + max_range / 2
    y_min, y_max = y_center - max_range / 2, y_center + max_range / 2

    # Set the figure size proportional to the ranges while maintaining the aspect ratio
    plt.figure(figsize=(max(5, max_range / 2), max(5, max_range / 2)))

    # Use multiples of 2 for grid intervals
    x_interval = y_interval = 2

    # Plot the first line with circles for points
    if x1 and y1:
        slope1 = (y1[-1] - y1[0]) / (x1[-1] - x1[0]) if (x1[-1] - x1[0]) != 0 else np.inf
        intercept1 = y1[0] - slope1 * x1[0]
        x_vals = np.array([x_min, x_max])  # Extend line to the plot limits
        plt.plot(x_vals, slope1 * x_vals + intercept1, color='black')

        # Plot circles at the points on the slanted line
        plt.scatter(x1, y1, color='black', s=100, zorder=5, edgecolor='black', label='Points on slanted line')

    # Plot the second line (horizontal)
    if x2 and y2:
        slope2 = (y2[-1] - y2[0]) / (x2[-1] - x2[0]) if (x2[-1] - x2[0]) != 0 else np.inf
        intercept2 = y2[0] - slope2 * x2[0]
        x_vals = np.array([x_min, x_max])  # Extend line to the plot limits
        plt.plot(x_vals, slope2 * x_vals + intercept2, color='black')

    # Set the aspect ratio of the plot to be square (1:1 aspect ratio)
    plt.gca().set_aspect('equal', adjustable='box')

    # Set the x and y axis limits
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    # Add gridlines with intervals of 2
    plt.grid(True, which='both', linestyle='--', linewidth=0.7)
    
    # Set ticks based on intervals of 2
    plt.xticks(np.arange(x_min, x_max, x_interval))
    plt.yticks(np.arange(y_min, y_max, y_interval))

    # Show the x and y axes
    plt.axhline(0, color='black', linewidth=1)
    plt.axvline(0, color='black', linewidth=1)

    # Remove the external tick labels but keep gridlines
    plt.gca().set_xticklabels([])
    plt.gca().set_yticklabels([])

    # Add labels for axes directly on the x and y axes without decimal points
    for i in np.arange(x_min, x_max, x_interval):
        if i != 0:
            plt.text(i, 0.5, f'{int(i)}', ha='center', va='center', fontsize=10, fontweight='bold')

    for i in np.arange(y_min, y_max, y_interval):
        if i != 0:
            plt.text(0.5, i, f'{int(i)}', ha='center', va='center', fontsize=10, fontweight='bold')

    # Save the plot to a BytesIO object and encode it in base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    plt.close()
    buffer.seek(0)

    return base64.b64encode(buffer.getvalue()).decode()

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

# Check if the prompt contains both "For the first line in the system" and "For the second line in the system"
if 'For the first line in the system' in full_prompt and 'For the second line in the system' in full_prompt:
    base64_graph = generate_two_linear_equation_graph(full_prompt)
    if base64_graph:
        question_info['Graph'] = f"data:image/png;base64,{base64_graph}"
        question_info['Stem Paragraph'] = "The graph of a system of linear equations is shown. What is the solution (x, y) to the system?"

# Check if the prompt contains points indicating a single linear equation
elif 'The line passes through the following points' in full_prompt:
    base64_graph = generate_single_linear_equation_graph(full_prompt)
    if base64_graph:
        question_info['Graph'] = f"data:image/png;base64,{base64_graph}"
        question_info['Stem Paragraph'] = "The graph of a linear equation is shown. What is the slope and y-intercept of the line?"

# Handle the case where a table is present
elif soup.find('table'):
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