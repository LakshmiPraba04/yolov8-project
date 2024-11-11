import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
data = pd.read_csv(r'C:\Users\laksh\Documents\sample\output\train3\results.csv')

# Ensure the data is loaded before proceeding
if not data.empty:
    # Strip unnecessary spaces from column names
    data.columns = data.columns.str.strip()

    # Plotting example (e.g., train/box_loss vs val/box_loss)
    plt.figure(figsize=(10, 5))
    plt.plot(data['epoch'], data['train/box_loss'], label='Train Box Loss')
    plt.plot(data['epoch'], data['val/box_loss'], label='Validation Box Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Box Loss')
    plt.title('Training vs Validation Box Loss')
    plt.legend()
    plt.show()
else:
    print("Error: The CSV file could not be loaded or is empty.")
