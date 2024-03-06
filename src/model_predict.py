import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os

def get_data(validation_datasize=5000):
    mnist = tf.keras.datasets.mnist
    (X_train_full, y_train_full), (X_test, y_test) = mnist.load_data()

    X_valid, X_train = X_train_full[:validation_datasize] / 255., X_train_full[validation_datasize:] / 255.
    y_valid, y_train = y_train_full[:validation_datasize], y_train_full[validation_datasize:]

    X_test = X_test / 255.

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)

# Assuming the get_data function has been called
_, _, (X_test, y_test) = get_data(validation_datasize=5000)

model = tf.keras.models.load_model('model/2024-03-06_123758_model.h5')

# Use testing_variable to specify the number of images to predict and display
testing_variable = 5
X_new = X_test[:testing_variable]
y_prob = model.predict(X_new)
Y_pred = np.argmax(y_prob, axis=-1)  # Get predictions

# Create a directory for the predicted images
predict_image_dir = "predicted_images"
os.makedirs(predict_image_dir, exist_ok=True)

# Plot and save all images in a single composite image
fig, axes = plt.subplots(1, testing_variable, figsize=(5 * testing_variable, 5))  # Adjusted figsize based on testing_variable
for i, (img_arr, pred, actual) in enumerate(zip(X_new, Y_pred, y_test[:testing_variable])):
    axes[i].imshow(img_arr, cmap="binary")
    axes[i].set_title(f"Predicted: {pred}, Actual: {actual}")
    axes[i].axis("off")

# Save the composite image
composite_image_path = os.path.join(predict_image_dir, "predictions_composite.png")
plt.savefig(composite_image_path)
plt.close(fig)  # Close the figure to free memory

print(f"Composite image saved at: {composite_image_path}")
