import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D
import matplotlib.pyplot as plt

# print(tf.config.list_physical_devices('GPU'))

# Define the directories where your data is stored
train_noisy_dir = 'C:\\Users\\xoryt\\OneDrive\\Documents\\info\\Hackatons\\ML&AI Electron\\train_noisy'
train_clear_dir = 'C:\\Users\\xoryt\\OneDrive\\Documents\\info\\Hackatons\\ML&AI Electron\\train'
test_noisy_dir = 'C:\\Users\\xoryt\\OneDrive\\Documents\\info\\Hackatons\\ML&AI Electron\\val_noisy'
test_clear_dir = 'C:\\Users\\xoryt\\OneDrive\\Documents\\info\\Hackatons\\ML&AI Electron\\val_noisy'

# def preprocess_image(image):
#     # Convert grayscale images to RGB format
#     if image.shape[-1] == 1:
#         image = tf.image.grayscale_to_rgb(image)
#     return image

# Create an ImageDataGenerator object
datagen = ImageDataGenerator(rescale=1./255)

# Load images in batches from the training directories
train_noisy_gen = datagen.flow_from_directory(
    train_noisy_dir,
    target_size=(400, 400),
    batch_size=32,
    class_mode=None)

train_clear_gen = datagen.flow_from_directory(
    train_clear_dir,
    target_size=(400, 400),
    batch_size=32,
    class_mode=None)

# Load images in batches from the testing directories
test_noisy_gen = datagen.flow_from_directory(
    test_noisy_dir,
    target_size=(400, 400),
    batch_size=32,
    class_mode=None)

test_clear_gen = datagen.flow_from_directory(
    test_clear_dir,
    target_size=(400, 400),
    batch_size=32,
    class_mode=None)

# Define the input shape
input_img = Input(shape=(400, 400, 3))

# Define the encoder
x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = MaxPooling2D((2, 2), padding='same')(x)
encoded = Conv2D(256, (3, 3), activation='relu', padding='same')(x)

# Define the decoder
x = Conv2D(256, (3, 3), activation='relu', padding='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
x = UpSampling2D((2, 2))(x)
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

# Create the autoencoder model
autoencoder = Model(input_img, decoded)

# Compile the model
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Define the model
# model = Sequential()
# model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
# model.add(MaxPooling2D((2, 2), padding='same'))
# model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
# model.add(Conv2D(3, (3, 3), activation='relu', padding='same'))
# model.add(UpSampling2D((2, 2)))

num_samples = len(train_noisy_gen.filenames)
batch_size = train_noisy_gen.batch_size
steps_per_epoch = num_samples // batch_size

# Compile the model
# model.compile(optimizer='adam', loss=combined_loss)

# images = next(train_noisy_gen)
# target = next(train_clear_gen)

# Train the model using the image generators
# model.fit(
#     x=images,
#     y=target,
#     batch_size=32,
#     steps_per_epoch=steps_per_epoch,
#     epochs=10)

def train_generator():
    while True:
        noisy_batch = next(train_noisy_gen)

        clear_batch = next(train_clear_gen)

        yield (noisy_batch, clear_batch)

autoencoder.fit(
    x=train_generator(),
    steps_per_epoch=3, #steps_per_epoch
    epochs=10)

# Use the model to denoise the images
denoised_images = autoencoder.predict(test_noisy_gen, steps=10)

# Predict on the first 10 test images.
for i in range(10):
    # Get the next image from the generator
    image = next(test_noisy_gen)
    
    # Predict the denoised image
    denoised_image = autoencoder.predict(image)
    print(denoised_image[0].shape)
    print(denoised_image[0].min())
    print(denoised_image[0].max())
    # denoised_image = (denoised_image - denoised_image.min()) / (denoised_image.max() - denoised_image.min())
    # denoised_image = denoised_image.astype('float32') / 255.0
    # Display the original (noisy) and denoised images
    fig, ax = plt.subplots(1, 2)
    
    ax[0].imshow(image[0])
    ax[0].set_title('Original')
    ax[0].axis('off')
    
    ax[1].imshow(denoised_image[0], cmap='gray')
    ax[1].set_title('Denoised')
    ax[1].axis('off')
    
    plt.show()