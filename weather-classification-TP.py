#!/usr/bin/env python
# coding: utf-8

import argparse
import os
from datetime import datetime
from glob import glob
import pandas as pd
import numpy as np
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.image import resize
from tensorflow.keras.models import load_model
from tqdm import tqdm

# Categories
class_names = {0: 'cloudy', 1: 'foggy', 2: 'rainy', 3: 'shine', 4: 'sunrise'}

# Function to load image
def load_image(path):
    """
    Loads an image from the given path and resizes it to 256x256.
    """
    try:
        img = resize(img_to_array(load_img(path)) / 255.0, (256, 256))
        return img
    except Exception as e:
        print(f"Error loading image {path}: {e}")
        return None

# Function to validate images
def validate_image(path):
    """
    Validates if the image file is readable.
    """
    from PIL import Image
    try:
        with Image.open(path) as img:
            img.verify()
        return True
    except Exception:
        return False

# Function to load already processed images
def load_processed_images(output_dir):
    """
    Loads processed image names from existing CSVs in the output directory.
    """
    processed_images = set()
    if not os.path.exists(output_dir):
        return processed_images

    for file in glob(os.path.join(output_dir, "*.csv")):
        try:
            df = pd.read_csv(file)
            processed_images.update(df["image_name"].tolist())
        except Exception as e:
            print(f"Error reading file {file}: {e}")
    return processed_images

# Main script
if __name__ == "__main__":
    # Argument parsing
    parser = argparse.ArgumentParser(description="Predict weather conditions from images.")
    parser.add_argument("--input_dir", required=True, help="Directory containing input images.")
    parser.add_argument("--output_dir", required=True, help="Directory to save predictions.")
    parser.add_argument("--force", action="store_true", help="Force prediction on all images, even if already processed.")
    args = parser.parse_args()

    # Validate input and output directories
    if not os.path.exists(args.input_dir):
        print(f"Input directory {args.input_dir} does not exist.")
        exit(1)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)

    # Load image paths and validate them
    image_paths = sorted(glob(f"{args.input_dir}/*.jpg"))
    image_paths = [path for path in image_paths if validate_image(path)]

    if not image_paths:
        print("No valid images found in input directory.")
        exit(1)

    print(f"Found {len(image_paths)} valid images.")

    # Skip already processed images if not in force mode
    if not args.force:
        processed_images = load_processed_images(args.output_dir)
        image_paths = [path for path in image_paths if os.path.basename(path) not in processed_images]
        print(f"Skipping {len(processed_images)} already processed images.")
        print(f"{len(image_paths)} images left for prediction.")

    if not image_paths:
        print("No new images to process.")
        exit(0)

    # Load images
    X = np.zeros((len(image_paths), 256, 256, 3))
    for i, path in tqdm(enumerate(image_paths), desc="Loading images"):
        img = load_image(path)
        if img is not None:
            X[i] = img

    # Load the pre-trained model
    try:
        model = load_model('ResNet152V2-Weather-Classification-03.h5')
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

    # Make predictions
    predictions = np.argmax(model.predict(X), axis=-1)

    # Save predictions to a CSV file
    results = pd.DataFrame({
        "image_name": [os.path.basename(path) for path in image_paths],
        "prediction_label": [class_names[p] for p in predictions]
    })

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(args.output_dir, f"predictions_{timestamp}.csv")
    
    try:
        results.to_csv(output_file, index=False)
        print(f"Predictions saved to {output_file}")
    except Exception as e:
        print(f"Error saving predictions to {output_file}: {e}")
