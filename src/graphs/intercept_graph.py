import numpy as np
import matplotlib.pyplot as plt
import base64
import io

# Adjust the function to remove bold edges from the axes
def intercept_graph(points, x_label, y_label, x_min, x_max, y_min, y_max, x_increment=1, y_increment=1, title=None, padding_factor=0.05):
    # Extract x and y coordinates from points
    points_x = [p[0] for p in points]
    points_y = [p[1] for p in points]

    # Create the parabola by fitting a polynomial to the points
    coefficients = np.polyfit(points_x, points_y, 2)
    parabola = np.poly1d(coefficients)

    # Generate x values for a smooth curve
    x_vals = np.linspace(x_min, x_max, 300)
    y_vals = parabola(x_vals)

    # Create the figure and axis
    fig, ax = plt.subplots()

    # Plot the parabola with a black line
    ax.plot(x_vals, y_vals, color='black')

    # Draw normal lines for x and y axes without bolding
    ax.axhline(0, color='black', linewidth=1)
    ax.axvline(0, color='black', linewidth=1)

    # Set the x and y limits to center the graph at (0, 0)
    x_padding_right = (x_max - x_min) * padding_factor
    y_padding_top = (y_max - y_min) * padding_factor
    ax.set_xlim(x_min, x_max + x_padding_right)
    ax.set_ylim(y_min - y_padding_top, y_max + y_padding_top)

    # Set custom ticks for the x and y axes using the provided increments
    ax.set_xticks(np.arange(x_min, x_max + 1, x_increment))
    ax.set_yticks(np.arange(y_min, y_max + 1, y_increment))

    # Place the labels on the axes instead of at the edges
    for label in ax.get_xticks():
        ax.text(label, -0.5, str(label), ha='center', va='center')

    for label in ax.get_yticks():
        ax.text(-0.5, label, str(label), ha='center', va='center')

    # Remove the default labels and title
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    if title:
        ax.set_title(title)

    # Add gridlines
    ax.grid(True)

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

# Example points from the parabola description
points = [(-1, -9), (0, -5), (1, -4.2)]

# Labels and limits for the axes
x_label = 'X axis'
y_label = 'Y axis'
x_min = -4
x_max = 4
y_min = -12
y_max = 0

# Generate the parabola graph with centered axes, normal x/y lines, and customizable increments
img_base64 = intercept_graph(points, x_label, y_label, x_min, x_max, y_min, y_max, x_increment=2, y_increment=2)

# Output the base64 image (or use it elsewhere)
img_base64