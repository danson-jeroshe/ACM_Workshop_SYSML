import os
import pickle
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path

# Base directory where your parts are stored
BASE_DIR = Path(r"C:\Users\daksh\OneDrive\Desktop\fedlearn\data\data\CIFAR10_dirichlet0.05_12")

NUM_CLIENTS = 12  # 12 parts

for client_id in range(NUM_CLIENTS):
    try:
        part_dir = BASE_DIR / f"part_{client_id}" / "CIFAR10_dirichlet0.05_12"
        data_path = part_dir / "train_data.pth"

        # Load train_data.pth file
        with open(data_path, "rb") as f:
            data = pickle.load(f)

        # Get labels
        try:
            labels = [sample['y'] for sample in data]
        except:
            labels = [sample[1] for sample in data]

        # Count label frequencies
        label_counts = Counter(labels)
        sorted_counts = dict(sorted(label_counts.items()))  # sorted by label

        # Print distribution info
        print(f"\n=== Client {client_id} ===")
        print(f"Total Samples: {len(labels)}")
        print("Label Counts:")
        for label, count in sorted_counts.items():
            print(f"  Label {label}: {count} samples")

        # Plot label distribution
        plt.figure(figsize=(6, 4))
        plt.bar(sorted_counts.keys(), sorted_counts.values(), color="skyblue")
        plt.title(f"Client {client_id} Label Distribution")
        plt.xlabel("Label")
        plt.ylabel("Count")
        plt.xticks(range(10))  # for CIFAR-10 labels
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.tight_layout()

        # Save the plot
        plt.savefig(f"client_{client_id}_label_distribution.png")
        plt.close()

    except Exception as e:
        print(f"[✗] Error processing client {client_id}: {e}")
