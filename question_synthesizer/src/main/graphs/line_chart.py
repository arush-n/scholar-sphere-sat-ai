import matplotlib.pyplot as plt
import numpy as np
import io
import base64

# DONE

def line_chart(groups, values, x_label, x_min, x_max, y_label, y_min, y_max, y_increment, x_increment=1, title=None, padding_factor=0.05):
    # Create the figure and axis
    fig, ax = plt.subplots()

    # Convert groups (years) to integers for better handling with x_increment
    groups = list(map(int, groups))
    
    # Create x positions based on x_min, x_max, and x_increment
    x_positions = np.arange(x_min, x_max + 1, x_increment)

    # Ensure the values align with the x_positions
    if len(x_positions) != len(values):
        raise ValueError("Number of x positions (groups) does not match the number of values.")

    # Create the line chart
    ax.plot(x_positions, values, color='black', marker='o')

    # Add labels for each point on the x-axis
    ax.set_xticks(x_positions)
    ax.set_xticklabels(groups)

    # Calculate padding for the right side of the x-axis and top of the y-axis
    x_padding_right = (x_max - x_min) * padding_factor
    y_padding_top = (y_max - y_min) * padding_factor

    # Apply padding to axis limits (slight padding on the left to prevent cutting off)
    x_min_padded = x_min - 0.5  # Slight padding on the left
    x_max_padded = x_max + x_padding_right  # Right side padding only
    y_max_padded = y_max + y_padding_top  # Top padding only

    # Set manual axis limits
    ax.set_xlim(x_min_padded, x_max_padded)  # Padding on left and right
    ax.set_ylim(y_min, y_max_padded)  # No padding on bottom

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
    print(img_base64)
    return img_base64

import re
from bs4 import BeautifulSoup

def line_chart_data(html_content: str):
    # Parse the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')

    # Step 1: Find the div with role="region" and aria-label containing "Long description for line graph"
    description_div = soup.find('div', {'role': 'region', 'aria-label': 'Long description for line graph'})
    
    if not description_div:
        raise ValueError("Long description for line graph not found")

    # Step 2: Find the <ul> inside this div (assuming it contains the groups and values)
    description_list = description_div.find('ul')
    if not description_list:
        raise ValueError("List of groups and values not found in long description")

    # Extract groups (years) and values from <li> elements within the <ul>
    groups = []
    values = []
    for li in description_list.find_all('li'):
        # Extract year (or group)
        group_match = re.search(r'(\d{4})', li.text)  # Capture 4-digit years
        # Extract value, either percentages or numbers
        value_match = re.search(r'(\d+)%', li.text)  # Capture numbers with or without the '%' symbol
        
        if group_match:
            groups.append(group_match.group(1))  # Append year as string
        if value_match:
            # Store the percentage value as an integer
            values.append(int(value_match.group(1)))

    # Validate groups and values to ensure proper extraction
    if not groups or not values:
        raise ValueError("Groups (years) or values (percentages) were not extracted correctly")

    # Step 3: Truncate the first element from the groups to match the values
    groups = groups[1:]
    values = values[1:]
    print(groups, values)
     
    # Ensure the lengths of groups and values match after truncation
    if len(groups) != len(values):
        raise ValueError("Mismatch between truncated groups and values.")

    # Step 4: Extract other necessary chart info like x_label, y_label, etc., from aria-label of the SVG element
    svg_element = soup.find('svg')
    if svg_element and svg_element.get('aria-label'):
        aria_label = svg_element['aria-label']
        
        # Example regex patterns for extracting x/y axis labels and ranges
        x_label_match = re.search(r'horizontal axis is labeled (\w+ \w+)', aria_label)
        x_label = x_label_match.group(1) if x_label_match else 'X Label'
        
        y_label_match = re.search(r'vertical axis is labeled (.+?)\.', aria_label)
        y_label = y_label_match.group(1) if y_label_match else 'Y Label'
        
        # Extract x_min, x_max, and x_increment from the aria-label
        x_min_match = re.search(r'It ranges from (\d+)', aria_label)
        x_min = int(x_min_match.group(1)) if x_min_match else 0
        
        x_max_match = re.search(r'to (\d+) in increments', aria_label)
        x_max = int(x_max_match.group(1)) if x_max_match else 100
        
        x_increment_match = re.search(r'in increments of (\d+)', aria_label)
        x_increment = int(x_increment_match.group(1)) if x_increment_match else 1

        # Extract y_min, y_max, and y_increment
        y_min = 0  # Based on the aria-label description: "It ranges from 0% to 15%"
        y_max_match = re.search(r'to (\d+)%', aria_label)
        y_max = int(y_max_match.group(1)) if y_max_match else 15
        
        y_increment_match = re.search(r'in increments of (\d+)', aria_label)
        y_increment = int(y_increment_match.group(1)) if y_increment_match else 1
        
        # Extract optional title (if mentioned in aria-label)
        title_match = re.search(r'A line graph. (.+?)\.', aria_label)
        title = title_match.group(1) if title_match else None
    else:
        raise ValueError("SVG with aria-label not found or aria-label is missing")

    # Step 5: Set padding factor (defaulting to 0.05 if not found elsewhere)
    padding_factor = 0.05

    
    # Step 6: Call the line_chart method with the extracted data and return it
    return line_chart(
        groups=groups,
        values=values,
        x_label=x_label,
        x_min=x_min,
        x_max=x_max,
        x_increment=x_increment,
        y_label=y_label,
        y_min=y_min,
        y_max=y_max,
        y_increment=y_increment,
        title=title,
        padding_factor=padding_factor
    )



with open("src\\testing\\test.html", "r", encoding="utf-8") as file:
    html_content = file.read()

chart_output = line_chart_data(html_content)    
# Example of how you might call this function
print(chart_output)

# 4a2264b3