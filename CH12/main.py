import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models, losses
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# load the Fashion-MNIST from the Keras module.
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
print(train_labels)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(train_images.shape)
print(test_images.shape)

plt.figure()
plt.imshow(train_images[42])
plt.colorbar()
plt.grid(False)
plt.title(class_names[train_labels[42]])
plt.show()

# we need to rescale the data to a range of 0 to 1
train_images = train_images / 255.0
test_images = test_images / 255.0

for i in range(16):
    plt.subplot(4, 4, i + 1)
    plt.subplots_adjust(hspace=.3)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.title(class_names[train_labels[i]])

plt.show()

X_train = train_images.reshape((train_images.shape[0], 28, 28, 1))
X_test = test_images.reshape((test_images.shape[0], 28, 28, 1))
print(X_train.shape)

tf.random.set_seed(42)

# initialize a Keras-based model
model = models.Sequential()
# 32 small-sized 3 * 3 filters.
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# max-pooling layer with a 2 * 2
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
# resulting filter maps are then flattened to provide features
model.add(layers.Flatten())
# hidden layer with 64 nodes:
model.add(layers.Dense(64, activation='relu'))
# output layer has 10 nodes representing 10 different classes
model.add(layers.Dense(10, activation='softmax'))
model.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])
model.summary()

model.fit(X_train, train_labels, validation_data=(X_test, test_labels), epochs=10)

test_loss, test_acc = model.evaluate(X_test, test_labels, verbose=2)
print('Accuracy on test set:', test_acc)

predictions = model.predict(X_test)
print(predictions[0])
print('Predicted label for the first test sample: ', np.argmax(predictions[0]))
print('True label for the first test sample: ', test_labels[0])


# plotting the sample image and the prediction results
def plot_image_prediction(i, images, predictions, labels, class_names):
    plt.subplot(1, 2, 1)
    plt.imshow(images[i], cmap=plt.cm.binary)
    prediction = np.argmax(predictions[i])
    color = 'blue' if prediction == labels[i] else 'red'
    plt.title(f"{class_names[labels[i]]} (predicted {class_names[prediction]})", color=color)
    plt.subplot(1,2,2)
    plt.grid(False)
    plt.xticks(range(10))
    plot = plt.bar(range(10), predictions[i], color="#777777")
    plt.ylim([0, 1])
    plot[prediction].set_color('red')
    plot[labels[i]].set_color('blue')
    plt.show()


plot_image_prediction(0, test_images, predictions, test_labels, class_names)

# Visualizing the convolutional filters
filters, _ = model.layers[2].get_weights()
f_min, f_max = filters.min(), filters.max()
n_filters = 16
for i in range(n_filters):
    filter = filters[:, :, :, i]
    plt.subplot(4, 4, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(filter[:, :, 0], cmap='gray')
plt.show()


# data augmentation
# datagen = ImageDataGenerator(horizontal_flip=True)
#
#
# def generate_plot_pics(datagen, original_img, save_prefix):
#     folder = 'aug_images'
#     i = 0
#     for batch in datagen.flow(original_img.reshape((1, 28, 28, 1)), batch_size=1, save_to_dir=folder, save_prefix=save_prefix, save_format='jpeg'):
#         i += 1
#         if i > 2:
#             break
#     plt.subplot(2, 2, 1, xticks=[],yticks=[])
#     plt.imshow(original_img)
#     plt.title("Original")
#     i = 1
#     for file in os.listdir(folder):
#         if file.startswith(save_prefix):
#             plt.subplot(2, 2, i + 1, xticks=[],yticks=[])
#             aug_img = load_img(folder + "/" + file)
#             plt.imshow(aug_img)
#             plt.title(f"Augmented {i}")
#             i += 1
#     plt.show()
#
#
# generate_plot_pics(datagen, train_images[0], 'horizontal_flip')
#
# datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
# generate_plot_pics(datagen, train_images[0], 'hv_flip')
#
# datagen = ImageDataGenerator(rotation_range=30)
# generate_plot_pics(datagen, train_images[0], 'rotation')
#
# datagen = ImageDataGenerator(width_shift_range=8)
# generate_plot_pics(datagen, train_images[0], 'width_shift')
#
# datagen = ImageDataGenerator(width_shift_range=8, height_shift_range=8)
# generate_plot_pics(datagen, train_images[0], 'width_height_shift')
#
# # Improving the clothing image classifier with data augmentation
# n_small = 500
# X_train = X_train[:n_small]
# train_labels = train_labels[:n_small]
# print(X_train.shape)
#
# model = models.Sequential()
# model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
# model.add(layers.MaxPooling2D((2, 2)))
# model.add(layers.Conv2D(64, (3, 3), activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dense(32, activation='relu'))
# model.add(layers.Dense(10, activation='softmax'))
#
# model.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])
# model.fit(X_train, train_labels, validation_data=(X_test, test_labels), epochs=20, batch_size=40)
#
# test_loss, test_acc = model.evaluate(X_test, test_labels, verbose=2)
# print('Accuracy on test set:', test_acc)
#
# datagen = ImageDataGenerator(height_shift_range=3, horizontal_flip=True)
#
# model_aug = tf.keras.models.clone_model(model)
#
# model_aug.compile(optimizer='adam', loss=losses.sparse_categorical_crossentropy, metrics=['accuracy'])
#
# train_generator = datagen.flow(X_train, train_labels, seed=42, batch_size=40)
# model_aug.fit(train_generator, epochs=50, validation_data=(X_test, test_labels))
#
# test_loss, test_acc = model_aug.evaluate(X_test, test_labels, verbose=2)
# print('Accuracy on test set:', test_acc)