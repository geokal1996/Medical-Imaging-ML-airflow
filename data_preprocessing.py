import os
import pydicom
import numpy as np
from datetime import datetime
import logging
from skimage.transform import resize

# Set up logging
logging.basicConfig(
    filename='preprocessing_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def preprocess_dicom(input_dir, output_dir):
    """Preprocess DICOM files by converting to NumPy arrays and saving."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for filename in os.listdir(input_dir):
        if filename.lower().startswith('anon_') and filename.lower().endswith('.dcm'):
            input_path = os.path.join(input_dir, filename)
            
            try:
                # Read DICOM file
                ds = pydicom.dcmread(input_path)
                logging.info(f"Processing {filename}: Shape {ds.pixel_array.shape}")
                print(f"Processing {filename}: Shape {ds.pixel_array.shape}")
                
                # Convert to NumPy array and normalize (e.g., to [0, 1])
                image_array = ds.pixel_array.astype(float)
                # Guard against a zero-range image (all pixels identical), which
                # would otherwise divide by zero and produce NaNs throughout the
                # array. Not observed in the CBIS-DDSM run this thesis reports on,
                # but a latent bug worth closing defensively. Added after the
                # results in Chapter 5 were finalized; it does not change them.
                value_range = image_array.max() - image_array.min()
                if value_range > 0:
                    image_array = (image_array - image_array.min()) / value_range
                else:
                    image_array = np.zeros_like(image_array)
                
                # Ensure 3D shape (add singleton dimension for 2D images)
                if len(image_array.shape) == 2:
                    image_array = np.expand_dims(image_array, axis=0)  # Add slice dimension
                elif len(image_array.shape) == 3:
                    pass  # Already 3D
                else:
                    raise ValueError(f"Unsupported shape for {filename}: {image_array.shape}")
                
                # Resize to a standard size (e.g., 11, 256, 256) - adjust for 3D data
                target_shape = (11, 256, 256)
                if image_array.shape[1:] != (256, 256):  # Check spatial dimensions
                    processed_array = resize(image_array, target_shape, mode='constant', preserve_range=True)
                else:
                    # Pad or trim slices to match target number
                    if image_array.shape[0] < target_shape[0]:
                        padding = np.zeros((target_shape[0] - image_array.shape[0], target_shape[1], target_shape[2]))
                        processed_array = np.vstack((image_array, padding))
                    elif image_array.shape[0] > target_shape[0]:
                        processed_array = image_array[:target_shape[0]]
                    else:
                        processed_array = image_array
                
                # Save as NumPy file with explicit .npy extension
                base_name = os.path.splitext(filename)[0]  # Get base name without extension
                output_path = os.path.join(output_dir, f"proc_{base_name}.npy")
                np.save(output_path, processed_array)
                logging.info(f"Saved processed {filename} to {output_path}")
                print(f"Saved processed {filename} to {output_path}")
            except Exception as e:
                logging.error(f"Error processing {filename}: {str(e)}")
                print(f"Error processing {filename}: {str(e)}")

def main(input_dir="data/anonymized_dicom", output_dir="data/processed_dicom"):
    """Main function to run data preprocessing."""
    if not os.path.exists(input_dir):
        print(f"Error: {input_dir} does not exist. Please run data ingestion first.")
        logging.error(f"Error: {input_dir} does not exist.")
        return

    start_time = datetime.now()
    logging.info(f"Starting preprocessing at {start_time}")
    print(f"Starting preprocessing at {start_time}")
    preprocess_dicom(input_dir, output_dir)
    logging.info(f"Preprocessing completed at {datetime.now()}. Total time: {datetime.now() - start_time}")
    print(f"Preprocessing completed at {datetime.now()}. Total time: {datetime.now() - start_time}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Convert anonymized DICOM files to preprocessed .npy volumes.")
    parser.add_argument("--input-dir", default="data/anonymized_dicom", help="Directory of anon_*.dcm files")
    parser.add_argument("--output-dir", default="data/processed_dicom", help="Directory to write proc_*.npy files")
    args = parser.parse_args()
    main(input_dir=args.input_dir, output_dir=args.output_dir)