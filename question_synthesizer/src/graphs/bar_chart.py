
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

def bar_chart(groups, values, x_label, y_label, y_min, y_max, y_increment, title=None, padding_factor=0.05):
    # Create the figure and axis
    fig, ax = plt.subplots()

    # Set positions for bars on the x-axis
    x_positions = np.arange(len(groups))

    # Create the bar chart
    ax.bar(x_positions, values, color='grey')

    # Add numerical labels for each bar
    ax.set_xticks(x_positions)
    ax.set_xticklabels(groups)


    # Calculate padding for the right side of the x-axis and top of the y-axis
    x_max = len(groups) - 1
    x_padding_right = (x_max - 0) * padding_factor
    y_padding_top = (y_max - y_min) * padding_factor

    # Apply padding to axis limits
    ax.set_xlim(-0.5, x_max + x_padding_right)  # Slight padding on both ends of the x-axis
    y_max_padded = y_max + y_padding_top  # Top padding only
    ax.set_ylim(y_min, y_max_padded)

    # Set the increments for the y-axis
    ax.set_yticks(np.arange(y_min, y_max + y_increment, y_increment))

    # Set the labels for the x-axis and y-axis
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Set an optional title if provided
    if title:
        ax.set_title(title)

    # Set grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Ensure white background for the plot
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Convert plot to base64 for further use (e.g., embedding in web pages)
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()

    # Return the base64 string
    return img_base64


import re
from bs4 import BeautifulSoup

def bar_chart_data(html_content: str):
    # Parse the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')

    # Step 1: Find the div with role="region" and aria-label containing "Long description for bar graph"
    description_div = soup.find('div', {'role': 'region', 'aria-label': 'Long description for bar graph'})
    
    if not description_div:
        raise ValueError("Long description for bar graph not found")

    # Step 2: Find the <ul> inside this div (assuming it contains the groups and values)
    description_list = description_div.find('ul')
    if not description_list:
        raise ValueError("List of groups and values not found in long description")

    # Extract groups (numeric categories) and values from <li> elements within the <ul>
    groups = []
    values = []
    
    for li in description_list.find_all('li'):
        # Extract group (numeric value)
        group_match = re.search(r'(\d+)', li.text)  # Capture the numeric value for the group
        # Extract value, which is a number
        value_match = re.search(r': (\d+)', li.text)  # Capture the values after ":"
        
        if group_match and value_match:
            groups.append(int(group_match.group(1)))  # Append the group number
            values.append(int(value_match.group(1)))  # Store the value as an integer

    # Validate groups and values to ensure proper extraction
    if not groups or not values:
        raise ValueError("Groups or values were not extracted correctly")

    # Step 3: Ensure the lengths of groups and values match
    if len(groups) != len(values):
        raise ValueError("Mismatch between groups and values.")

    # Step 4: Extract other necessary chart info like x_label, y_label, etc., from aria-label of the SVG element
    svg_element = soup.find('svg')
    if svg_element and svg_element.get('aria-label'):
        aria_label = svg_element['aria-label']
        
        # Extract x_label and y_label based on the specific part in aria-label
        x_label_match = re.search(r'horizontal axis is labeled ([\w\s]+)\.', aria_label)
        x_label = x_label_match.group(1) if x_label_match else 'X Label'
        
        y_label_match = re.search(r'vertical axis is labeled ([\w\s]+)\.', aria_label)
        y_label = y_label_match.group(1) if y_label_match else 'Y Label'
        
        # Extract y_min, y_max, and y_increment
        y_min_match = re.search(r'vertical axis ranges from (\d+)', aria_label)
        y_min = int(y_min_match.group(1)) if y_min_match else 0
        
        y_max_match = re.search(r'to (\d+)', aria_label)
        y_max = int(y_max_match.group(1)) if y_max_match else max(values)
        
        y_increment_match = re.search(r'in increments of (\d+)', aria_label)
        y_increment = int(y_increment_match.group(1)) if y_increment_match else 10
        
        # Extract optional title (if mentioned in aria-label)
        title_match = re.search(r'A bar chart. (.+?)\.', aria_label)
        title = title_match.group(1) if title_match else None
    else:
        raise ValueError("SVG with aria-label not found or aria-label is missing")

    # Step 5: Set padding factor (defaulting to 0.05 if not found elsewhere)
    padding_factor = 0.05
    groups = groups[1:]
    values = values[1:]
    
    # Step 6: Return the data in a structured format (or call a method to create the bar chart if needed)
    return bar_chart(
        groups=groups,
        values=values,
        x_label=x_label,  # Use extracted x_label
        y_label=y_label,  # Use extracted y_label
        y_min=y_min,
        y_max=y_max,
        y_increment=y_increment,
        title=title,
        padding_factor=padding_factor
    )


with open("src\\testing\\test.html", "r", encoding="utf-8") as file:
    html_content = file.read()

chart_output = bar_chart_data(html_content)    
# Example of how you might call this function
print(chart_output)