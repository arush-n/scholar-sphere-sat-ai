import matplotlib.pyplot as plt
import numpy as np
import io
import base64
from scipy.interpolate import make_interp_spline

def curve_chart(groups, values, x_label, y_label, y_min, y_max, y_increment, title=None, padding_factor=0.05):
    # Create the figure and axis
    fig, ax = plt.subplots()

    # Set positions for the points on the x-axis
    x_positions = np.arange(len(groups))

    # Create smooth curve between the points
    x_smooth = np.linspace(x_positions.min(), x_positions.max(), 300)
    spline = make_interp_spline(x_positions, values)
    y_smooth = spline(x_smooth)

    # Create the curve chart
    ax.plot(x_smooth, y_smooth, color='black')

    # Mark the original data points
    ax.scatter(x_positions, values, color='black', zorder=5)

    # Add labels for each point
    ax.set_xticks(x_positions)
    ax.set_xticklabels(groups)

    # Calculate padding for the right side of the x-axis and top of the y-axis
    x_padding_right = (x_positions.max() - x_positions.min()) * padding_factor
    y_padding_top = (y_max - y_min) * padding_factor

    # Apply padding to axis limits
    x_min_padded = x_positions.min() - 0.5  # Slight padding on the left
    x_max_padded = x_positions.max() + x_padding_right  # Right side padding only
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
    return img_base64

