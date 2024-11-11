import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
data = pd.read_csv(r'C:\Users\laksh\Documents\sample\output\train3\results.csv')

# Ensure the data is loaded before proceeding
if not data.empty:
    # Strip unnecessary spaces from column names
    data.columns = data.columns.str.strip()

    # Plotting Precision and Recall
    plt.figure(figsize=(10, 5))
    plt.plot(data['epoch'], data['metrics/precision(B)'], label='Precision')
    plt.plot(data['epoch'], data['metrics/recall(B)'], label='Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Precision and Recall over Epochs')
    plt.legend()
    plt.show()

    # Plotting mAP
    plt.figure(figsize=(10, 5))
    plt.plot(data['epoch'], data['metrics/mAP50(B)'], label='mAP50')
    plt.plot(data['epoch'], data['metrics/mAP50-95(B)'], label='mAP50-95')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.title('mAP over Epochs')
    plt.legend()
    plt.show()

else:
    print("Error: The CSV file could not be loaded or is empty.")
