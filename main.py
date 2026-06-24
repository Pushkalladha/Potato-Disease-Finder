import matplotlib
matplotlib.use('Agg')

import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers

IMAGE_SIZE = 256
BATCH_SIZE = 32

# Load dataset
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset",
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE
)

# Class names
class_names = dataset.class_names

print("Classes:", class_names)

# Dataset split
dataset_size = len(dataset)

train_size = int(0.8 * dataset_size)
val_size = int(0.1 * dataset_size)

train_ds = dataset.take(train_size)
val_ds = dataset.skip(train_size).take(val_size)
test_ds = dataset.skip(train_size + val_size)

# Optimization
AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)

val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Data augmentation
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal_and_vertical"),
    tf.keras.layers.RandomRotation(0.2),
])
resize_and_rescale = tf.keras.Sequential([
    layers.Resizing(256,256),
    layers.Rescaling(1./255)
])
model = models.Sequential([
    layers.Input(shape=(256,256,3)),

    resize_and_rescale,
    data_augmentation,

    layers.Conv2D(32,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64,(3,3),activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Flatten(),

    layers.Dense(64, activation='relu'),

    layers.Dense(3, activation='softmax')
])

print("REACHED MODEL")

model.summary()



model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("REACHED COMPILE")


print("Starting training...")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50
)

model.save("potato_disease_model.keras")
print("Model saved successfully!")


# Visualize augmented images
for images, labels in train_ds.take(1):

    augmented_images = data_augmentation(images)

    plt.figure(figsize=(10,10))

    for i in range(9):

        ax = plt.subplot(3,3,i+1)

        plt.imshow(augmented_images[i].numpy().astype("uint8"))

        plt.title(class_names[labels[i]])

        plt.axis("off")

    plt.savefig("final_pipeline_output.png")

