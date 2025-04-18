import numpy as np
import matplotlib.pyplot as plt
from sympy import symbols, Eq, solve
import base64
import io

def plot_rational_fx(point1, point2, va, ha, x_min, x_max, y_min, y_max, x_increment, y_increment):
    """
    This function plots the rational function of the form:
    f(x) = ha + A(x + B) / -(x - va)
    for given constants A, B, vertical asymptote va, and horizontal asymptote ha.
    The plot uses manually provided x_min, x_max, y_min, y_max values and allows for custom x and y increments for ticks.
    """
    # Define the rational function
    def rational_function_from_points(point1, point2, va, ha):
        """
        This function calculates and returns the rational function in its general form
        based on two points and given vertical and horizontal asymptotes.
        """
        # Define variables
        A, B, x = symbols('A B x')
        
        # Extract point coordinates
        x1, y1 = point1
        x2, y2 = point2
        
        # Set up the equations from the two points
        eq1 = Eq(y1, ha + A * (x1 + B) / -(x1 - va))
        eq2 = Eq(y2, ha + A * (x2 + B) / -(x2 - va))
        
        # Solve for A and B
        solution = solve((eq1, eq2), (A, B))
        
        # Extract values of A and B from solution
        A_val, B_val = solution[0]
        
        # Construct the final rational function f(x) = ha + A(x + B)/-(x - va)
        return lambda x: ha + A_val * (x + B_val) / -(x - va)

    # Define the rational function using the points and asymptotes
    rational_function = rational_function_from_points(point1, point2, va, ha)

    # Generate values for x (excluding the vertical asymptote at x = va)
    x_values = np.linspace(x_min, va - 0.01, 400)
    x_values_right = np.linspace(va + 0.01, x_max, 400)

    # Compute y-values for the function
    y_values = rational_function(x_values)
    y_values_right = rational_function(x_values_right)

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot the function with a black curve
    ax.plot(x_values, y_values, color='black')
    ax.plot(x_values_right, y_values_right, color='black')

    # Bold the x and y axes
    ax.spines['left'].set_linewidth(2)
    ax.spines['bottom'].set_linewidth(2)

    # Remove borders on the top and right sides
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

    # Position the x and y axes at the center (zero)
    ax.spines['left'].set_position('zero')
    ax.spines['bottom'].set_position('zero')

    # Add gridlines
    ax.grid(True)

    # Set manual axis limits
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # Set custom tick increments
    ax.set_xticks(np.arange(x_min, x_max + x_increment, x_increment))
    ax.set_yticks(np.arange(y_min, y_max + y_increment, y_increment))

    # Convert plot to base64 for further use (e.g., embedding in web pages)
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight', facecolor='white')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()

    # Return the base64 string
    return img_base64

# Example use: Solve for A and B and plot the function using two points and asymptotes with manual limits and tick increments
point1 = (-3, 2)
point2 = (0, 3)
va = 4
ha = 1
x_min = -10
x_max = 10
y_min = -10
y_max = 10
x_increment = 2
y_increment = 1

img_base64 = plot_rational_fx(point1, point2, va, ha, x_min, x_max, y_min, y_max, x_increment, y_increment)

img_base64