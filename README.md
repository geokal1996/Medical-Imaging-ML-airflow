# Automated Machine Learning Workflow for Medical Imaging with Apache Airflow

Reference implementation for the MSc thesis *Automation of Machine Learning Workflows in
the Medical Domain Using Apache Airflow* (MSc in Medical Physics – Radiation Physics,
University of Patras, 2026).

A three-stage pipeline — DICOM ingestion and anonymization, preprocessing, and model
training — orchestrated as a linear Airflow DAG, together with the evaluation protocol
used to produce the results reported in the thesis.

---

## What this is, and what it is not

**It is** a working, end-to-end automated workflow for medical imaging data, verified
through a real logged DAG execution, plus a reproducible evaluation protocol for small
datasets.

**It is not** a clinically validated diagnostic tool. The case study uses 80 images.
The trained classifier reaches a mean test ROC-AUC of 0.694 — above chance, but far
from anything clinically meaningful. See [Limitations](#limitations).

---

## Pipeline

| Stage | Script | What it does |
|---|---|---|
| 1. Ingestion | `data_ingestion.py` | Walks the input tree recursively, strips identifying DICOM tags, writes anonymized copies |
| 2. Preprocessing | `data_preprocessing.py` | Converts anonymized DICOM to normalized NumPy arrays of fixed shape `(11, 256, 256)` |
| 3. Training | `model_training.py` | Trains a 3D CNN and saves the model |
| Orchestration | `medical_workflow_dag.py` | Wires the three stages into an Airflow DAG: `process_dicom >> preprocess_data >> train_model` |
| Evaluation | `evaluate.py` | Stratified 60/20/20 split, five independent training runs, reports mean ± std |

Stages communicate through the filesystem: each reads a directory and writes another.
Every stage also exposes an `argparse` CLI, so it can be run and tested standalone.

---

## A note on anonymization

`anonymize_dicom()` removes four standard patient tags — PatientName `(0010,0010)`,
PatientID `(0010,0020)`, PatientBirthDate `(0010,0030)`, PatientSex `(0010,0040)` — and
then calls `remove_private_tags()`.

That second call exists because of a finding during development. An earlier version of
the ingestion module removed only the standard tags. Inspecting the *output* files showed
that patient identifiers were still recoverable: a manufacturer-specific private tag
block retained PatientID and date of birth after the standard tags had been deleted.
That earlier module was superseded and is not included here.

**The practical lesson: verify de-identification on the actual output, rather than
trusting the procedure.**

This is a targeted fix, not full de-identification. It does **not** implement the DICOM
PS3.15 Annex E confidentiality profile, which covers many more attributes (institution
names, study dates, accession numbers, UIDs), and it does **not** handle identifiers
burned into pixel data. Do not treat it as a compliance solution.

---

## Data

The case study uses a balanced 80-image subset of the **CBIS-DDSM** mammography
collection (40 benign / 40 malignant, one image per patient, no patient overlap between
classes), obtained from [The Cancer Imaging Archive](https://www.cancerimagingarchive.net/).

**The images are not distributed here.** `cbis_ddsm_selection.csv` lists the exact case
identifiers and labels used, so the same subset can be reproduced from TCIA.

TCIA downloads nest each `.dcm` file several directories deep under generated names, and
the `PatientID` tag was found to be inconsistently formatted for roughly half the files.
Ingestion therefore derives the case identifier from the top-level case folder, enabled
with `--use-case-folder-as-filename`.

---

## Running it

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Stages standalone:

```bash
python data_ingestion.py --raw-dir data/raw_dicom --anon-dir data/anonymized_dicom \
                         --use-case-folder-as-filename
python data_preprocessing.py --input-dir data/anonymized_dicom \
                             --output-dir data/processed_dicom
python model_training.py --input-dir data/processed_dicom \
                         --label-csv cbis_ddsm_selection.csv
```

Full evaluation (five runs, writes `evaluation/metrics.txt` and plots):

```bash
python evaluate.py --input-dir data/cbis_ddsm_processed \
                   --label-csv cbis_ddsm_selection.csv --n-runs 5
```

Under Airflow, place `medical_workflow_dag.py` in a **dedicated** DAG folder — pointing
`dags_folder` at a directory that also contains a virtual environment makes the DAG
processor attempt to import every installed library. Then point the DAG at the pipeline
scripts:

```bash
export MEDICAL_PIPELINE_DIR=/path/to/medical_ml_pipeline
airflow standalone
```

---

## Results

Held-out test set (16 images), mean ± standard deviation across five independent
training runs on a fixed stratified split:

| Metric | Mean ± Std |
|---|---|
| Accuracy | 0.6375 ± 0.1075 |
| Precision | 0.6567 ± 0.1269 |
| Recall | 0.6000 ± 0.1458 |
| F1-score | 0.6196 ± 0.1214 |
| ROC-AUC | 0.6937 ± 0.0794 |

Full per-run figures: `evaluation/metrics.txt`.

### Why five runs rather than one

Fixing `random_state` and calling `tf.random.set_seed()` was **not** sufficient for
reproducibility. Three runs of the identical architecture on the identical split produced
test accuracies of 0.625, 0.75 and 0.6875. TensorFlow's CPU convolution ops use
multi-threaded reductions whose floating-point accumulation order is not deterministic,
and eliminating that would require disabling intra-op parallelism entirely.

Rather than chase single-run determinism, the protocol measures the variance: train
`n_runs` times on the same fixed split, varying only the training seed, and report mean
and standard deviation. On a 16-image test set a single number is one draw from a
distribution, not a result.

---

## Limitations

- **Sample size.** 80 images total, 48 for training. Every other limitation follows from this.
- **Model capacity.** ~31.5M trainable parameters, of which 31,490,176 sit in a single
  `Dense(128)` layer after `Flatten()`. Replacing it with `GlobalAveragePooling3D`
  (64,705 parameters) was tested and performed worse in a single exploratory run
  (ROC-AUC 0.844 → 0.359), possibly because global averaging discards the spatial
  localization that small calcifications depend on. One run is not enough to establish
  that as the cause.
- **3D architecture on 2D inputs.** Mammograms are 2D; each is expanded to a synthetic
  volume. The depth axis carries no independent information. A 2D-native architecture
  would be more appropriate for this dataset.
- **Preprocessing inconsistency.** Images already at 256×256 are zero-padded in the slice
  dimension while others are interpolated across all three axes at once.
- **Variance scope.** The reported standard deviation reflects training stochasticity
  only, not variability across different patient partitions. No cross-validation.
- **No early stopping, no hyperparameter search.** Training runs a fixed 10 epochs;
  hyperparameters were set manually.
- **Deployment.** The pipeline ends at a saved model file. No serving layer, no API,
  no monitoring, no clinical system integration.

---

## Citation

```bibtex
@mastersthesis{kalokairinos2026,
  author = {Kalokairinos, Georgios},
  title  = {Automation of Machine Learning Workflows in the Medical Domain
            Using Apache Airflow},
  school = {University of Patras},
  year   = {2026},
  type   = {MSc thesis}
}
```

## License

MIT — see [LICENSE](LICENSE).
