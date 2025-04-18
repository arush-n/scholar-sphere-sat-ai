
# id - 1adb39f0
import matplotlib.pyplot as plt
import numpy as np
import io
import base64

def scatterplot(x_points, y_points, x_label, y_label, x_min, x_max, y_min, y_max, x_increment, y_increment, line_start=None, line_end=None, padding_factor=0.025, title=None):
    # Create the figure and axis
    fig, ax = plt.subplots()

    # Plot the scatter points
    ax.scatter(x_points, y_points, color='black')

    # Draw a manual line that extends to the edges of the graph, not influencing the scaling
    if line_start and line_end:
        # Calculate the slope and intercept of the line
        slope = (line_end[1] - line_start[1]) / (line_end[0] - line_start[0])
        intercept = line_start[1] - slope * line_start[0]
        
        # Extend the line to cover the full x-range (from x_min to x_max)
        x_extended = [x_min, x_max]
        y_extended = [slope * x_min + intercept, slope * x_max + intercept]
        
        # Plot the extended line
        ax.plot(x_extended, y_extended, color='black')

    # Calculate padding for both x and y axes
    x_range = x_max - x_min
    y_range = y_max - y_min

    # Adjust padding to ensure that points on the axis are shown, but no extra increment appears
    x_padding = min((x_increment / 2), x_range * padding_factor)
    y_padding = min((y_increment / 2), y_range * padding_factor)

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

# Example usage
x_points = [0, 0.8, 1, 1.5, 2, 2.2, 3, 3.5, 3.8, 4]
y_points = [12, 11, 12, 10.3, 10, 9.2, 9, 8, 8.3, 7]
x_label = 'x'
y_label = 'y'
x_min = 0
x_max = 6
y_min = 6
y_max = 12
x_increment = 1
y_increment = 1


# Manual line points (for a manual line of best fit or any other line)
line_start = (1, 11)
line_end = (4, 7.5)
# Generate the graph and get the base64 string
img_base64 = scatterplot(x_points, y_points, x_label, y_label, x_min, x_max, y_min, y_max, x_increment, y_increment, line_start=line_start, line_end=line_end)

# Print the base64 string (or use it elsewhere)
print(img_base64)
