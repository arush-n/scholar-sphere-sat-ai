import re
import numpy as np
import matplotlib as plt
import re
import base64
import io

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