
import csv
import os
import pickle
import math
import random
from collections import Counter

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "dataset", "student_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

# Read dataset
X = []
y = []

with open(DATA_PATH, "r", newline="") as file:
    reader = csv.DictReader(file)

    for row in reader:
        features = [
            float(row["attendance"]) / 100,
            float(row["study_hours"]) / 24,
            float(row["previous_marks"]) / 100,
            float(row["assignments"]) / 100
        ]

        X.append(features)
        y.append(row["performance"])


# KNN prediction algorithm
def predict_knn(train_X, train_y, sample, k=3):

    distances = []

    for features, label in zip(train_X, train_y):

        distance = math.sqrt(
            sum((a - b) ** 2 for a, b in zip(features, sample))
        )

        distances.append((distance, label))

    distances.sort(key=lambda item: item[0])

    nearest = [label for _, label in distances[:k]]

    return Counter(nearest).most_common(1)[0][0]


# Split data for testing
random.seed(42)

indices_by_class = {}

for i, label in enumerate(y):
    indices_by_class.setdefault(label, []).append(i)

test_indices = []

for indices in indices_by_class.values():
    random.shuffle(indices)
    test_indices.append(indices[0])

train_indices = [
    i for i in range(len(X)) if i not in test_indices
]

train_X = [X[i] for i in train_indices]
train_y = [y[i] for i in train_indices]

correct = 0

for i in test_indices:
    prediction = predict_knn(train_X, train_y, X[i])
    if prediction == y[i]:
        correct += 1

accuracy = correct / len(test_indices) * 100

print("KNN Model Training Successful!")
print("Test Accuracy:", round(accuracy, 2), "%")


# Train final model using complete dataset
model = {
    "algorithm": "K-Nearest Neighbors",
    "X": X,
    "y": y,
    "k": 3
}

MODEL_PATH = os.path.join(
    MODEL_DIR, "student_performance_model.pkl"
)

with open(MODEL_PATH, "wb") as file:
    pickle.dump(model, file)

print("Final model saved successfully!")
print("Model location:", MODEL_PATH)
