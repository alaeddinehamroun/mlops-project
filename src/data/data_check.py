# Class imbalance
import os
import matplotlib.pyplot as plt

# Set the path to your dataset directory
dataset_dir = 'PokemonData'

# Initialize a dictionary to store the count of images per class
class_counts = {}

# Iterate through the subfolders in the dataset directory
for class_name in os.listdir(dataset_dir):
    class_dir = os.path.join(dataset_dir, class_name)
    if os.path.isdir(class_dir):
        # Count the number of images in each class
        num_images = len(os.listdir(class_dir))
        class_counts[class_name] = num_images

# Visualize the class distribution
class_labels = list(class_counts.keys())
class_images = list(class_counts.values())

plt.figure(figsize=(10, 6))
plt.bar(class_labels, class_images)
plt.xlabel('Pokémon Class')
plt.ylabel('Number of Images')
plt.title('Class Distribution of Pokémon Images')
plt.xticks(rotation=90)
plt.show()

# Compute class imbalance metrics
total_images = sum(class_images)
class_frequencies = {class_label: count / total_images for class_label, count in class_counts.items()}
class_imbalance_ratio = max(class_frequencies.values()) / min(class_frequencies.values())

print('Class Frequencies:')
for class_label, frequency in class_frequencies.items():
    print(f'{class_label}: {frequency:.2%}')

print('Class Imbalance Ratio:', class_imbalance_ratio)
