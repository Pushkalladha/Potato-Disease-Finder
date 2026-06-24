import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

IMAGE_SIZE = 256
BATCH_SIZE = 32

# Load dataset
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset",
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE
)

class_names = dataset.class_names

dataset_size = len(dataset)

train_size = int(0.8 * dataset_size)
val_size = int(0.1 * dataset_size)

test_ds = dataset.skip(train_size + val_size)

# Load trained model
model = tf.keras.models.load_model("potato_disease_model.keras")

# Evaluate
loss, accuracy = model.evaluate(test_ds)

print(f"Test Accuracy: {accuracy*100:.2f}%")

# Show predictions
plt.figure(figsize=(12,12))

for images, labels in test_ds.take(1):

    predictions = model.predict(images)

    for i in range(9):

        actual = class_names[labels[i]]
        predicted = class_names[np.argmax(predictions[i])]
        confidence = np.max(predictions[i]) * 100

        plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(
            f"A:{actual}\nP:{predicted}\nC:{confidence:.2f}%"
        )
        plt.axis("off")

plt.tight_layout()
plt.savefig("prediction_results.png")

print("Saved prediction_results.png")