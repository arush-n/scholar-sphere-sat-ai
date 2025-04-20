import matplotlib.pyplot as plt
import numpy as np
import io
import base64

def parabolic_scatterplot(x_label, y_label, x_min, x_max, y_min, y_max, x_increment, y_increment, points, title=None, padding_factor=0.05):
    # Extract the given points
    left_high = points.get('left_high')
    left_low = points.get('left_low')
    right_low = points.get('right_low')
    right_high = points.get('right_high')

    # Generate 12 x-values between the left_high x-value and the right_high x-value
    x_points = np.linspace(left_high[0], right_high[0], 12)

    # Use a quadratic interpolation to generate y-values based on the given key points
    x_key = [left_high[0], left_low[0], right_low[0], right_high[0]]
    y_key = [left_high[1], left_low[1], right_low[1], right_high[1]]

    # Use np.polyfit to fit a parabola through the 4 key points
    parabola_coeff = np.polyfit(x_key, y_key, 2)
    y_points = np.polyval(parabola_coeff, x_points)

    # Create the figure and axis
    fig, ax = plt.subplots()

    # Plot the scatter points
    ax.scatter(x_points, y_points, color='black')

    # Calculate padding for both x and y axes
    x_range = x_max - x_min
    y_range = y_max - y_min

    x_padding = x_range * padding_factor
    y_padding = y_range * padding_factor

    # Apply padding to axis limits
    x_min_padded = x_min - x_padding
    x_max_padded = x_max + x_padding
    y_min_padded = y_min - y_padding
    y_max_padded = y_max + y_padding

    # Set manual axis limits with padding
    ax.set_xlim(x_min_padded, x_max_padded)
    ax.set_ylim(y_min_padded, y_max_padded)

    # Set the increments for the x and y axes
    ax.set_xticks(np.arange(x_min, x_max + x_increment, x_increment))
    ax.set_yticks(np.arange(y_min, y_max + y_increment, y_increment))

    # Set an optional title if provided
    if title:
        ax.set_title(title)

    # Dynamic labels for axes
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Set grid and ensure black and white
    ax.grid(True)
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Convert plot to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()

    # Return the base64 string
    return img_base64

# Example key points for an upward-opening parabola
points = {
    'left_high': (0, 20),    # Left high point
    'left_low': (2.5, 5),    # Left low point
    'right_low': (3, 5),     # Right low point
    'right_high': (5.5, 20)  # Right high point
}

# Axis labels and limits
x_label = 'x'
y_label = 'y'
x_min = 0
x_max = 6
y_min = 0
y_max = 20
x_increment = 1
y_increment = 2

# Optional title
title = "Dynamic Parabolic Scatter Plot"

# Generate the graph and get the base64 string
img_base64 = parabolic_scatterplot(x_label, y_label, x_min, x_max, y_min, y_max, x_increment, y_increment, points, title=title)

# Print the base64 string (or use it elsewhere)
print(img_base64)
