import os
import shutil
import pydicom
from datetime import datetime
import logging


# Set up logging
logging.basicConfig(
    filename='ingestion_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def anonymize_dicom(ds):
    """Anonymize DICOM file by removing sensitive tags."""

    sensitive_tags = [
        (0x0010, 0x0010),  # PatientName
        (0x0010, 0x0020),  # PatientID
        (0x0010, 0x0030),  # PatientBirthDate
        (0x0010, 0x0040),  # PatientSex
    ]

    for tag in sensitive_tags:
        if tag in ds:
            del ds[tag]

    # Remove private manufacturer-specific tags.
    ds.remove_private_tags()

    return ds


def ingest_dicom_files(
    input_dir,
    output_dir,
    use_case_folder_as_filename=False,
    case_folder_prefix="Calc-Test_"
):
    """Ingest, anonymize, and save DICOM files."""

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for root, _, files in os.walk(input_dir):
        for filename in files:

            if not filename.lower().endswith('.dcm'):
                continue

            input_path = os.path.join(root, filename)

            try:
                # Read DICOM file
                ds = pydicom.dcmread(input_path)

                logging.info(f"Processing DICOM file: {filename}")
                print(f"Processing DICOM file: {filename}")

                if use_case_folder_as_filename:
                    rel = os.path.relpath(root, input_dir)

                    case_folder = (
                        rel.split(os.sep)[0]
                        if rel != '.'
                        else os.path.splitext(filename)[0]
                    )

                    if case_folder.lower().endswith('.dcm'):
                        case_folder = case_folder[:-4]

                    if (
                        case_folder_prefix
                        and not case_folder.startswith(case_folder_prefix)
                    ):
                        case_folder = f"{case_folder_prefix}{case_folder}"

                    base_name = case_folder

                else:
                    base_name = os.path.splitext(filename)[0]

                output_path = os.path.join(
                    output_dir,
                    f"anon_{base_name}.dcm"
                )

                # Anonymize DICOM dataset
                ds = anonymize_dicom(ds)

                # Save anonymized DICOM file
                ds.save_as(output_path)

                logging.info(
                    f"Moved and anonymized {filename} to {output_path}"
                )

                print(
                    f"Moved and anonymized {filename} to {output_path}"
                )

            except pydicom.errors.InvalidDicomError:
                logging.warning(
                    f"Skipping invalid DICOM file: {filename}"
                )

                print(
                    f"Skipping invalid DICOM file: {filename}"
                )

            except Exception as e:
                logging.error(
                    f"Error processing {filename}: {str(e)}"
                )

                print(
                    f"Error processing {filename}: {str(e)}"
                )


def main(
    raw_dir="data/raw_dicom",
    anon_dir="data/anonymized_dicom",
    use_case_folder_as_filename=False
):
    """Main function to run data ingestion."""

    if not os.path.exists(raw_dir):
        print(
            f"Error: {raw_dir} does not exist. "
            "Please add DICOM files to this directory."
        )

        logging.error(
            f"Error: {raw_dir} does not exist."
        )

        return

    start_time = datetime.now()

    logging.info(
        f"Starting ingestion at {start_time}"
    )

    print(
        f"Starting ingestion at {start_time}"
    )

    ingest_dicom_files(
        raw_dir,
        anon_dir,
        use_case_folder_as_filename=use_case_folder_as_filename
    )

    logging.info(
        f"Ingestion completed at {datetime.now()}. "
        f"Total time: {datetime.now() - start_time}"
    )

    print(
        f"Ingestion completed at {datetime.now()}. "
        f"Total time: {datetime.now() - start_time}"
    )


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Anonymize DICOM files recursively "
            "into an output directory."
        )
    )

    parser.add_argument(
        "--raw-dir",
        default="data/raw_dicom",
        help="Input directory containing .dcm files"
    )

    parser.add_argument(
        "--anon-dir",
        default="data/anonymized_dicom",
        help="Output directory for anonymized files"
    )

    parser.add_argument(
        "--use-case-folder-as-filename",
        action="store_true",
        help=(
            "Name output files from the top-level "
            "case subfolder instead of the input filename"
        )
    )

    args = parser.parse_args()

    main(
        raw_dir=args.raw_dir,
        anon_dir=args.anon_dir,
        use_case_folder_as_filename=args.use_case_folder_as_filename
    )