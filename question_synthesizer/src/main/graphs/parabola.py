import matplotlib.pyplot as plt
import numpy as np
import io
import base64
# Updated function to plot parabola with gridlines and black line, without labels and key
def parabola(points, x_label, y_label, x_min, x_max, y_min, y_max, title=None, padding_factor=0.05):
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


    # Remove labels for each point

    # Calculate padding for the right side of the x-axis and top of the y-axis
    x_padding_right = (x_max - x_min) * padding_factor
    y_padding_top = (y_max - y_min) * padding_factor

    # Apply padding to axis limits
    x_min_padded = x_min
    x_max_padded = x_max + x_padding_right  # Right side padding only
    y_max_padded = y_max + y_padding_top  # Top padding only

    # Set the x and y limits
    ax.set_xlim(x_min_padded, x_max_padded)
    ax.set_ylim(y_min, y_max_padded)

    # Remove axis labels and title
    ax.set_xlabel('')
    ax.set_ylabel('')
    if title:
        ax.set_title('')

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
points = [(0.0, 3.8), (0.6, 5.5), (1.0, 4.8), (1.7, 0.0)]

# Labels and limits for the axes
x_label = 'Time, in seconds'
y_label = 'Height above ground, in meters'
x_min = 0
x_max = 3
y_min = 0
y_max = 7

# Generate the parabola graph with gridlines and no labels
img_base64 = parabola(points, x_label, y_label, x_min, x_max, y_min, y_max)

# Print the base64 string (or use it elsewhere)
print(img_base64)
