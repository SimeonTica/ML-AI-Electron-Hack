import argparse
from keras.models import load_model
from PIL import Image
import numpy as np
import os


# Create the parser
parser = argparse.ArgumentParser(description="Run inference")

# Add the arguments
parser.add_argument('--input_path', type=str, required=True, help='Path to the input data')
parser.add_argument('--output_path', type=str, required=True, help='Path to save the output')

# Parse the arguments
args = parser.parse_args()

# print(f"Input path: {args.input_path}")
# print(f"Output path: {args.output_path}")

image_files = [f for f in os.listdir(args.input_path) if f.endswith('.jpg') or f.endswith('.png')]

model = load_model('autoencoder_model_best.keras')

# Load the images and convert them into numpy arrays
images = []
original_sizes = []
for f in image_files:
    img = Image.open(os.path.join(args.input_path, f))
    original_sizes.append(img.size)  # save the original size
    img_resized = img.resize((400, 400))
    images.append(np.array(img_resized) / 255.0)

# Convert the list of images into a single numpy array
data = np.array(images)

# Use the model to make predictions
predictions = model.predict(data)

def calculate_psnr(img1, img2):
    # Ensure the images are float type
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    # Calculate MSE
    mse = np.mean((img1 - img2)**2)

    # If MSE is zero, PSNR is infinity
    if mse == 0:
        return float('inf')

    # Calculate MAX_I
    max_i = np.max(img1)

    # Calculate PSNR
    return 20 * np.log10(max_i/(mse**0.5))

# Save the predictions as images
mean = 0
count = 0
for i, prediction in enumerate(predictions):
    # Rescale the prediction values to the range 0-255
    # Convert the prediction to an image
    prediction_img = Image.fromarray((prediction * 255).astype('uint8'))
    prediction_img = prediction_img.resize(original_sizes[i])
    mean += calculate_psnr(data[i], prediction)
    count+=1
    # Save the image
    prediction_img.save(os.path.join(args.output_path, f'prediction_{i}.jpg'))

print(f"Mean PSNR: {mean/count}")