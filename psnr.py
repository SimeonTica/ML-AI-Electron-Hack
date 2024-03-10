# Import necessary libraries
import os
import cv2
import argparse
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm

def calculate_average_psnr(folder1_path, folder2_path):
    """
    Calculates the average PSNR between images in two folders, ensuring images have the same resolution and data type.
    
    Args:
    - folder1_path: Path to the first folder containing images.
    - folder2_path: Path to the second folder containing images.
    
    Returns:
    - The average PSNR value for all image pairs.
    """
    folder1_files = sorted(os.listdir(folder1_path))
    folder2_files = sorted(os.listdir(folder2_path))

    if len(folder1_files) != len(folder2_files):
        raise ValueError("The folders contain a different number of files.")

    total_psnr = 0

    for file1, file2 in tqdm(zip(folder1_files, folder2_files), total=len(folder1_files), desc="Calculating PSNR"):
        image1_path = os.path.join(folder1_path, file1)
        image2_path = os.path.join(folder2_path, file2)

        image1 = cv2.imread(image1_path)
        image2 = cv2.imread(image2_path)

        if image1 is None or image2 is None:
            raise IOError(f"Error loading images: {file1} or {file2}")

        # Check if images have the same shape and type
        if image1.shape != image2.shape or image1.dtype != image2.dtype:
            raise ValueError(f"Images {file1} and {file2} do not have the same resolution or data type.")

        psnr_value = psnr(image1, image2)
        total_psnr += psnr_value    

    average_psnr = total_psnr / len(folder1_files)
    return average_psnr

def main():
    parser = argparse.ArgumentParser(description="Calculate the average PSNR between images in two folders.")
    parser.add_argument("folder1_path", type=str, help="Path to the first folder containing images.")
    parser.add_argument("folder2_path", type=str, help="Path to the second folder containing images.")
    args = parser.parse_args()

    try:
        average_psnr = calculate_average_psnr(args.folder1_path, args.folder2_path)
        print(f"Average PSNR: {average_psnr} dB")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
