import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# Helper function to load an image
def load_image(image_path):
    try:
        image = Image.open(image_path)
        return np.array(image)
    except FileNotFoundError:
        return np.array([0])

# Shuffle and normalize data
def shuffle_and_normalize(X, y):
    indices = np.random.permutation(len(X))
    X_shuffled = X[indices] / 255.0  # Normalize
    y_shuffled = y[indices]
    return X_shuffled, y_shuffled


# Load dataset metadata from CSV
cards_df = pd.read_csv("cards.csv")
print(cards_df.head())
print(f"Num of rows and columns: {cards_df.shape}")

# Map class indices to class names
num_classes = 53
labels_map = {
    i: str(cards_df.loc[cards_df["class index"] == i, "labels"].iloc[0])
    for i in range(num_classes)
}
print(f"Labels map: {labels_map}")

# Load and split image data into sets
X_train, y_train = [], []
X_val, y_val = [], []
X_test, y_test = [], []

for _, row in cards_df.iterrows():
    curr_class = int(row["class index"])
    curr_path = str(row["filepaths"])
    curr_set = str(row["data set"])
    curr_image = load_image(curr_path)

    if np.ndim(curr_image) > 1:
        if curr_set == "train":
            X_train.append(curr_image)
            y_train.append(curr_class)
        elif curr_set == "valid":
            X_val.append(curr_image)
            y_val.append(curr_class)
        elif curr_set == "test":
            X_test.append(curr_image)
            y_test.append(curr_class)

# Convert to numpy arrays
X_train, y_train = np.array(X_train), np.array(y_train)
X_val, y_val = np.array(X_val), np.array(y_val)
X_test, y_test = np.array(X_test), np.array(y_test)

# Visualize class distribution
plt.figure()
plt.hist(y_train, bins=num_classes)
plt.title("Histogram of training classes")
plt.show()

# Print dataset shapes
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

X_train, y_train = shuffle_and_normalize(X_train, y_train)
X_val, y_val = shuffle_and_normalize(X_val, y_val)
X_test, y_test = shuffle_and_normalize(X_test, y_test)

# Display a sample image
sample_idx = 7000

if sample_idx < len(X_train):
    plt.figure()
    plt.imshow(X_train[sample_idx])
    plt.title(f"Sample Image - Label: {labels_map[y_train[sample_idx]]}")
    plt.show()

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(512, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train, y_train,
    epochs=4,
    validation_data=(X_val, y_val)
)

# Evaluate on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f'Test accuracy: {test_acc:.4f}')

# Save the model
model.save('your_model.h5')

# Plot training history
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Training vs Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Training vs Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Make predictions and evaluate
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Accuracy check
accuracy = np.mean(y_pred == y_test) * 100
print(f"Prediction Accuracy: {accuracy:.2f}%")

# Display confusion matrix
# cm = confusion_matrix(y_test, y_pred)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm)
# disp.plot()
# plt.show()
