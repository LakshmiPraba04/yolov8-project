import pandas as pd
import matplotlib.pyplot as plt

# Load the data from the CSV file (update the file path as needed)
data = pd.read_csv('C:\\Users\\laksh\\Documents\\sample\\output\\train3\\results.csv')

# Strip any leading or trailing spaces from column names
data.columns = data.columns.str.strip()

# Print the column names to verify that spaces have been removed
print("Columns in DataFrame:", data.columns)

# Inspect the first few rows of the DataFrame to ensure it's loaded correctly
print(data.head())

# Plot the data
try:
    plt.plot(data['epoch'], data['metrics/mAP50(B)'], label='mAP50')
    plt.xlabel('Epoch')
    plt.ylabel('mAP50')
    plt.title('mAP50 vs. Epoch')
    plt.legend()
    plt.show()
except KeyError as e:
    print(f"Error: Column not found - {e}")
