
import matplotlib.pyplot as plt
import io
import base64

# question id to look @ ID: c4ea43ef
def line_graph_interpretation(hours_at_job_a, hours_at_job_b, x_label, y_label):
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the graph with black line and markers
    ax.plot(hours_at_job_a, hours_at_job_b, marker='o', color='black')

    # Dynamic labels for axes
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    # Set dynamic ticks for x and y axes based on max values
    ax.set_xticks(range(0, max(hours_at_job_a) + 1, 5))
    ax.set_yticks(range(0, max(hours_at_job_b) + 1, 5))

    # Adding the grid
    ax.grid(True)

    # Ensure the plot is black and white
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

# Example usage with dynamic data and labels
hours_at_job_a = [0, 16]  # Number of hours at job A
hours_at_job_b = [10, 0]  # Number of hours at job B
x_label = 'Number of hours at job A'
y_label = 'Number of hours at job B'

# Generate the graph and get the base64 string
img_base64 = line_graph_interpretation(hours_at_job_a, hours_at_job_b, x_label, y_label)

# Print base64 string (or use it elsewhere)
print(img_base64)