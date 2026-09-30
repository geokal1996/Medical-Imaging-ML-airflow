import os

import csv

import argparse

import numpy as np

from datetime import datetime

import logging

import tensorflow as tf

from tensorflow.keras import layers, models


# Set up logging

logging.basicConfig(
    filename='training_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_data(input_dir, label_dict=None, return_filenames=False):

    """Load preprocessed NumPy files and prepare data with customizable labels.

    return_filenames: when True, also returns the list of .npy filenames in
    the same order as data_array/labels. Off by default so existing callers
    (main() below) are unaffected; evaluate.py uses this to trace which
    files ended up in each train/val/test split.
    """

    data_list = []

    filenames = []

    for filename in os.listdir(input_dir):

        if filename.endswith('.npy'):

            file_path = os.path.join(input_dir, filename)

            data = np.load(file_path)

            data_list.append(data)

            filenames.append(filename)

    if not data_list:

        logging.error("No data found.")

        raise ValueError(
            "No preprocessed data files (.npy) found in the input directory."
        )

    data_array = np.array(data_list)

    # Default dummy labels; replace with label_dict for real data

    if label_dict is None:

        labels = np.array([
            1 if 'abnormal' in filename.lower() else 0
            for filename in filenames
        ])

        if len(set(labels)) == 1:

            logging.warning(
                f"Only {len(data_array)} sample(s) found with uniform dummy labels. "
                "Consider adding real labels."
            )

            print(
                f"Warning: Only {len(data_array)} sample(s) found with uniform "
                "dummy labels. Consider adding real labels."
            )

    else:

        # label_dict.get(..., 0) silently defaults unmatched files to label 0,
        # which would quietly mislabel samples if preprocessing output names
        # and the label source (CSV or hardcoded dict) ever drift apart. Warn
        # loudly instead of failing silently.

        missing = [
            f for f in filenames
            if f not in label_dict
        ]

        if missing:

            logging.warning(
                f"{len(missing)} file(s) have no entry in label_dict "
                f"and will default to label 0: {missing}"
            )

            print(
                f"Warning: {len(missing)} file(s) have no label match "
                f"in label_dict, defaulting to 0: {missing}"
            )

        labels = np.array([
            label_dict.get(filename, 0)
            for filename in filenames
        ])

    if return_filenames:

        return data_array, labels, filenames

    return data_array, labels


def load_labels_from_csv(csv_path):

    """Build a {npy_filename: label} dict from a case-selection CSV.

    Expects a CSV with at least 'folder_id' and 'label' columns, as produced
    when selecting a TCIA subset (e.g. cbis_ddsm_selection.csv): folder_id is
    the DICOM PatientID / TCIA case identifier, label is 0/1. Matches against
    the .npy filenames produced by data_preprocessing.py, which are named
    "proc_anon_{folder_id}.npy" when data_ingestion.py is run with
    use_case_folder_as_filename=True.
    """

    label_dict = {}

    with open(csv_path, newline='') as f:

        reader = csv.DictReader(f)

        for row in reader:

            folder_id = row['folder_id'].strip()

            label = int(row['label'])

            npy_filename = f"proc_anon_{folder_id}.npy"

            label_dict[npy_filename] = label

    return label_dict


def create_model(input_shape):

    """Create a 3D CNN model."""

    model = models.Sequential([

        layers.Input(shape=input_shape),

        layers.Conv3D(
            32,
            kernel_size=(3, 3, 3),
            activation='relu'
        ),

        layers.MaxPooling3D(
            pool_size=(2, 2, 2)
        ),

        layers.Conv3D(
            64,
            kernel_size=(3, 3, 3),
            activation='relu'
        ),

        layers.MaxPooling3D(
            pool_size=(2, 2, 2)
        ),

        layers.Flatten(),

        layers.Dense(
            128,
            activation='relu'
        ),

        layers.Dropout(0.5),

        layers.Dense(
            1,
            activation='sigmoid'
        )
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_model(
    model,
    x_train,
    y_train,
    epochs=10,
    batch_size=2,
    validation_split=0.2
):

    """Train the model with optional validation split."""

    if len(x_train) < 2:

        history = model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size
        )

        logging.warning(
            "Insufficient data for validation split. "
            "Trained on full dataset."
        )

        print(
            "Warning: Insufficient data for validation split. "
            "Trained on full dataset."
        )

    else:

        history = model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split
        )

    return history


def main(
    input_dir="data/processed_dicom",
    output_dir="models",
    label_csv=None
):

    """Main function to run model training.

    label_csv: optional path to a CSV with 'folder_id'/'label' columns
    (see load_labels_from_csv). When given, labels are loaded from this
    file instead of the hardcoded demo-dataset dictionary below.
    """

    if not os.path.exists(input_dir):

        print(
            f"Error: {input_dir} does not exist. "
            "Please run data preprocessing first."
        )

        logging.error(
            f"Error: {input_dir} does not exist."
        )

        return

    if not os.path.exists(output_dir):

        os.makedirs(output_dir)

    start_time = datetime.now()

    logging.info(
        f"Starting training at {start_time}"
    )

    print(
        f"Starting training at {start_time}"
    )

    if label_csv:

        label_dict = load_labels_from_csv(label_csv)

        logging.info(
            f"Loaded {len(label_dict)} labels from {label_csv}"
        )

        print(
            f"Loaded {len(label_dict)} labels from {label_csv}"
        )

    else:

        # Hardcoded labels for the original 6-file demo dataset.
        # These are NOT real clinical ground truth - they were assigned
        # arbitrarily just to make the demo pipeline runnable.
        # Kept only for backward compatibility when running with no --label-csv.

        label_dict = {

            'proc_anon_0003.npy': 0,  # Normal

            'proc_anon_0004.npy': 0,  # Normal

            'proc_anon_0009.npy': 1,  # Abnormal

            'proc_anon_0015.npy': 0,  # Normal

            'proc_anon_0020.npy': 1,  # Abnormal

            'proc_anon_MRBRAIN.npy': 0  # Normal
        }

    # Load data

    try:

        x_data, y_data = load_data(
            input_dir,
            label_dict
        )

    except ValueError as e:

        print(e)

        return

    # Reshape for 3D CNN (add channel dimension)

    x_data = x_data.reshape(
        (
            x_data.shape[0],
            x_data.shape[1],
            x_data.shape[2],
            x_data.shape[3],
            1
        )
    )

    input_shape = x_data.shape[1:]  # e.g., (11, 256, 256, 1)

    # Create and train model

    model = create_model(input_shape)

    logging.info(
        f"Model summary: {model.summary()}"
    )

    print(
        model.summary()
    )

    history = train_model(
        model,
        x_data,
        y_data
    )

    model.save(
        os.path.join(
            output_dir,
            f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5"
        )
    )

    logging.info(
        f"Model saved to {output_dir}"
    )

    print(
        f"Model saved to {output_dir}"
    )

    logging.info(
        f"Training completed at {datetime.now()}. "
        f"Total time: {datetime.now() - start_time}"
    )

    print(
        f"Training completed at {datetime.now()}. "
        f"Total time: {datetime.now() - start_time}"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train the 3D CNN on preprocessed DICOM data."
    )

    parser.add_argument(
        "--input-dir",
        default="data/processed_dicom",
        help="Directory of preprocessed .npy files"
    )

    parser.add_argument(
        "--output-dir",
        default="models",
        help="Directory to save the trained model"
    )

    parser.add_argument(
        "--label-csv",
        default=None,
        help=(
            "Optional CSV with folder_id,label columns "
            "(e.g. cbis_ddsm_selection.csv)"
        )
    )

    args = parser.parse_args()

    main(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        label_csv=args.label_csv
    )