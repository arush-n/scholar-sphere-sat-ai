import re
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

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
