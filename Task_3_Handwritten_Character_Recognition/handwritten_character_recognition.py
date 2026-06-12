import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

# Load Dataset
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

# Reshape
x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))

print("Dataset Loaded Successfully!")

# CNN Model
model = models.Sequential([
    layers.Input(shape=(28, 28, 1)),
    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compile
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("Training Started...")

# Train
model.fit(
    x_train,
    y_train,
    epochs=5,
    validation_data=(x_test, y_test)
)

# Evaluate
loss, accuracy = model.evaluate(x_test, y_test)

print("\n===== HANDWRITTEN CHARACTER RECOGNITION =====")
print("Model Accuracy:", accuracy)

# Save Model
model.save("handwritten_model.keras")
print("Model Saved Successfully!")

# ----------------------------
# IMAGE PREDICTION PART
# ----------------------------

image_path = "Image/test.png"

print("\nChecking File...")
print("Path:", image_path)
print("Exists:", os.path.exists(image_path))

if os.path.exists(image_path):

    saved_model = load_model("handwritten_model.keras")

    img = Image.open(image_path).convert("L")
    img = img.resize((28, 28))

    img_array = np.array(img)

    # Invert Colors
    img_array = 255 - img_array

    # Normalize
    img_array = img_array / 255.0

    # Reshape
    img_array = img_array.reshape(1, 28, 28, 1)

    prediction = saved_model.predict(img_array)

    predicted_digit = np.argmax(prediction)

    print("\n===== IMAGE PREDICTION =====")
    print("Predicted Digit:", predicted_digit)

else:

    print("\nERROR: test.png not found!")
    print("Place image here:")
    print("Task_3_Handwritten_Character_Recognition/Image/test.png")

print("\nProject Completed Successfully!")