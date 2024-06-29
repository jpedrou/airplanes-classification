import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy


# ==========================================================
# load data
# ==========================================================

data = tf.keras.utils.image_dataset_from_directory("../data/raw")
data_iterator = data.as_numpy_iterator()

batch = data_iterator.next()

# Images represented as numpy arrays
batch[0].shape

# Class label
# 0 = Commercial airplaine
# 1 = Military airplaine
batch[1]

fig, ax = plt.subplots(ncols=4, figsize=(20, 20))
for i, img in enumerate(batch[0][:4]):
    ax[i].imshow(img.astype(int))
    ax[i].title.set_text(batch[1][i])

# ==========================================================
# Preprocess data
# ==========================================================

# Scale data
data = data.map(lambda x, y: (x / 255, y))
data.as_numpy_iterator().next()[0].min(), data.as_numpy_iterator().next()[0].max()

# split data
train_size = int(len(data) * 0.7)
val_size = int(len(data) * 0.2) + 1
test_size = int(len(data) * 0.1)

print("Train Size:", train_size)
print("Validation Size:", val_size)
print("Test Size:", test_size)

train_size + val_size + test_size == len(data)

train = data.take(train_size)
validation = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)

len(train), len(validation), len(test)

# ==========================================================
# Deep model (CNN)
# ==========================================================

# Build Deep Learning Model
model = Sequential()

model.add(Conv2D(16, (3, 3), 1, activation="relu", input_shape=(256, 256, 3)))
model.add(MaxPooling2D())

model.add(Conv2D(32, (3, 3), 1, activation="relu"))
model.add(MaxPooling2D())

model.add(Conv2D(16, (3, 3), 1, activation="relu"))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(256, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile("adam", loss=tf.losses.BinaryCrossentropy(), metrics=["accuracy"])

model.summary()

# Train the model
hist = model.fit(train, epochs=20, validation_data=validation)

hist.history

# Plot performance

# loss plot
fig = plt.figure()
plt.plot(hist.history["loss"], color="red", label="loss")
plt.plot(hist.history["val_loss"], color="blue", label="val_loss")
fig.suptitle("CNN Loss", fontsize=20)
plt.legend(loc="upper left")
plt.show()

fig.savefig("../reports/figures/model-loss.jpg")

# Accuracy plot
fig = plt.figure()
plt.plot(hist.history["accuracy"], color="red", label="accuracy")
plt.plot(hist.history["val_accuracy"], color="blue", label="val_accuracy")
fig.suptitle("CNN Accuracy", fontsize=20)
plt.legend(loc="upper left")
plt.show()

fig.savefig("../reports/figures/model-accuracy.jpg")

# ==========================================================
# Evaluate performance
# ==========================================================

# Evaluate
precision = Precision()
recall = Recall()
accuracy = BinaryAccuracy()

len(test)

for batch in test.as_numpy_iterator():
    X, y = batch
    y_pred = model.predict(X)
    precision.update_state(y, y_pred)
    recall.update_state(y, y_pred)
    accuracy.update_state(y, y_pred)

print(
    f"Precision: {precision.result()}, Recall: {recall.result()}, Accuracy: {accuracy.result()}"
)

# ==========================================================
# Testing
# ==========================================================

# Load new images
img = cv2.imread("../test_images/commerical-airplaine-test.jpg")
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

# Resize image
resize = tf.image.resize(img, (256, 256))
plt.imshow(resize.numpy().astype(int))
plt.show()

y_pred = model.predict(np.expand_dims(resize / 255, 0))

if y_pred > 0.5:
    print("It's a military airplaine")
else:
    print("It's a commercial airplaine")

# ==========================================================
# Save the model
# ==========================================================

model.save("../model.h5")

new_model = load_model("../model.h5")
y_pred = new_model.predict(np.expand_dims(resize / 255, 0))

if y_pred > 0.5:
    print("It's a military airplaine")
else:
    print("It's a commercial airplaine")

