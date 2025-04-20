import matplotlib.pyplot as plt
import io
import base64

# Function to visualize the table using matplotlib and return as base64-encoded image
def stimulius_table(headers, rows, title):
    num_columns = len(headers)
    num_rows = len(rows)

    # Adjust figure size dynamically based on the number of rows and columns
    fig_width = max(6, num_columns * 1.5)  # Minimum width of 6, scaled by number of columns with extra space
    fig_height = max(3.5, num_rows * 0.7)  # Minimum height of 3.5, scaled by number of rows with extra space
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))  # Set dynamic size based on content
    ax.axis('tight')
    ax.axis('off')

    # Create the table with centered text
    table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center', colColours=["#f8f8f8"] * num_columns)

    # Set table title
    plt.title(title, fontsize=12, weight='bold')

    # Set the font size for all cells
    table.auto_set_font_size(False)
    table.set_fontsize(12)  # Adjust font size to fit text

    # Set padding using scale to ensure text fits properly inside cells
    table.scale(1.2, 1.5)  # Adjust scaling for padding (width, height)

    # Adjust table layout by setting equal width for all columns and height for all rows
    for i in range(num_rows + 1):  # Including header row
        for j in range(num_columns):  # Ensure all columns are processed
            try:
                cell = table[(i, j)]
                cell.set_fontsize(12)  # Set uniform font size
                cell.set_text_props(ha='center', va='center')  # Ensure centered text alignment
            except KeyError:
                print(f"Skipping cell ({i}, {j}) due to KeyError")

    # Save the figure to a BytesIO object and encode it in base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    plt.close()  # Close the plot to free up memory
    buffer.seek(0)

    # Encode the image in base64 and return it
    return base64.b64encode(buffer.getvalue()).decode()

# Example usage
headers = ["", "Amount invested", "Balance increase"]
rows = [
    ["Account A", "$500", "6% annual interest"],
    ["Account B", "$1,000", "$25 per year"]
]
title = "Investment Summary"

# Get the base64 encoded image of the table
image_base64 = stimulius_table(headers, rows, title)

# Print the base64 string (or use it for further processing)
print(image_base64)
