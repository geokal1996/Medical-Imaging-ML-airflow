"""Rigorous evaluation of the 3D CNN on a stratified train/val/test split.

Why this is a separate script from model_training.py: model_training.py is
the simple, automated pipeline stage (used standalone and later from the
Airflow DAG) - it trains on the whole dataset with an internal Keras
validation_split, which is fine for demonstrating that the pipeline runs,
but leaves no data the model has never been evaluated against. For results
that go in the thesis, this script instead:

  1. Splits the dataset into train/val/test (default 60/20/20) using a
     stratified split (same benign/malignant ratio in every split) with a
     fixed random_state, so the split is reproducible and documented. This
     split is held FIXED across all repeated runs below (see point 2).
  2. Trains the model n_runs times (default 5) on that same train/val split,
     varying only the weight initialization / batch shuffling seed between
     runs. This exists because a first attempt at full reproducibility -
     seeding random_state and calling tf.random.set_seed() for a single run
     - was tried on 2026-07-07 and found NOT sufficient: three runs of the
     identical architecture on the identical data split produced test
     accuracy 0.625, 0.75, and 0.6875 (ROC-AUC 0.84, 0.75, 0.75). TensorFlow's
     CPU convolution ops use multi-threaded reductions whose floating-point
     accumulation order is not fully deterministic even with every seed set,
     short of disabling intra-op parallelism entirely (which would make
     training far slower for a marginal, not-fully-guaranteed determinism
     gain). Rather than chase single-run determinism, this script instead
     embraces the variance: it runs training n_runs times and reports the
     mean and standard deviation of each metric. This is both more honest
     (a single run's number is one draw from a distribution, not "the"
     result) and standard practice for small-sample evaluations like this
     one (16 held-out test images).
  3. Evaluates each run exclusively on the held-out test set - data the
     model has never seen in any form during training - and reports
     accuracy, precision, recall, F1, and ROC-AUC per run, plus mean+-std
     across runs.
  4. Saves training curves for the last run, a SUMMED confusion matrix
     across all runs (how many times each true/predicted combination
     occurred, out of n_runs x 16 total predictions), and a metrics.txt with
     the full per-run table plus aggregate statistics, so the numbers used
     in the thesis are fully traceable back to one script invocation.

Reuses load_data / load_labels_from_csv / create_model from model_training.py
rather than duplicating that logic.
"""

import os
import argparse
import logging
import random
from datetime import datetime

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')  # no display available when run from the command line
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from model_training import load_data, load_labels_from_csv, create_model

logging.basicConfig(
    filename='evaluation_log.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def split_dataset(x_data, y_data, filenames, test_frac=0.2, val_frac=0.2, random_state=42):
    """Stratified train/val/test split with a fixed random_state for reproducibility.

    Splits off test_frac first, then splits the remainder into train/val so
    that val ends up as val_frac of the ORIGINAL data (not of the remainder).
    With the defaults (20%/20%) this gives a 60/20/20 train/val/test split.
    stratify=y_data keeps the benign/malignant ratio consistent across all
    three sets, which matters given how few samples we have (80 total).
    """
    x_temp, x_test, y_temp, y_test, f_temp, f_test = train_test_split(
        x_data, y_data, filenames,
        test_size=test_frac, stratify=y_data, random_state=random_state
    )
    val_frac_of_temp = val_frac / (1 - test_frac)
    x_train, x_val, y_train, y_val, f_train, f_val = train_test_split(
        x_temp, y_temp, f_temp,
        test_size=val_frac_of_temp, stratify=y_temp, random_state=random_state
    )
    return (x_train, y_train, f_train), (x_val, y_val, f_val), (x_test, y_test, f_test)


def plot_training_curves(history, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].plot(history.history['loss'], label='train loss')
    axes[0].plot(history.history['val_loss'], label='val loss')
    axes[0].set_title('Loss per epoch (last run)')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    axes[1].plot(history.history['accuracy'], label='train accuracy')
    axes[1].plot(history.history['val_accuracy'], label='val accuracy')
    axes[1].set_title('Accuracy per epoch (last run)')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def plot_confusion_matrix(cm, output_path, class_names=('benign', 'malignant'),
                           title='Confusion matrix (summed across runs)'):
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title(title)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                     color='white' if cm[i, j] > cm.max() / 2 else 'black')

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main(input_dir="data/cbis_ddsm_processed", label_csv="cbis_ddsm_selection.csv",
         output_dir="evaluation", epochs=10, batch_size=2, random_state=42, n_runs=5):
    os.makedirs(output_dir, exist_ok=True)

    start_time = datetime.now()
    logging.info(f"Starting evaluation run at {start_time}")
    print(f"Starting evaluation run at {start_time}")

    label_dict = load_labels_from_csv(label_csv)
    x_data, y_data, filenames = load_data(input_dir, label_dict, return_filenames=True)
    x_data = x_data.reshape((x_data.shape[0], x_data.shape[1], x_data.shape[2], x_data.shape[3], 1))

    # Split ONCE and hold it fixed across all n_runs repetitions below, so
    # the reported variance reflects only training randomness (weight init,
    # batch order) - not which images happened to land in the test set.
    (x_train, y_train, f_train), (x_val, y_val, f_val), (x_test, y_test, f_test) = split_dataset(
        x_data, y_data, filenames, random_state=random_state
    )
    print(f"Split: {len(f_train)} train / {len(f_val)} val / {len(f_test)} test "
          f"(train positives: {int(y_train.sum())}/{len(y_train)}, "
          f"val positives: {int(y_val.sum())}/{len(y_val)}, "
          f"test positives: {int(y_test.sum())}/{len(y_test)})")
    logging.info(f"Split sizes - train:{len(f_train)} val:{len(f_val)} test:{len(f_test)}")
    logging.info(f"Test set files: {f_test}")

    input_shape = x_data.shape[1:]

    run_rows = []
    summed_cm = np.zeros((2, 2), dtype=int)
    last_history = None
    last_model = None
    last_report = None

    for run_idx in range(n_runs):
        run_seed = random_state + run_idx
        random.seed(run_seed)
        np.random.seed(run_seed)
        tf.random.set_seed(run_seed)
        # Clear the Keras/TensorFlow backend session before building each new
        # model so state or memory from the previous run's model does not
        # accumulate across the n_runs loop. Added after the results in
        # Chapter 5 were finalized; since a fresh model was already built each
        # iteration via create_model(), this does not change those results -
        # it only makes repeated runs safer/cleaner going forward.
        tf.keras.backend.clear_session()

        model = create_model(input_shape)
        if run_idx == 0:
            print(model.summary())

        print(f"\n--- Run {run_idx + 1}/{n_runs} (seed={run_seed}) ---")
        history = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
        )

        y_prob = model.predict(x_test).ravel()
        y_pred = (y_prob >= 0.5).astype(int)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError as e:
            auc = float('nan')
            logging.warning(f"Run {run_idx + 1}: could not compute ROC-AUC: {e}")
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        summed_cm += cm

        row = dict(run=run_idx + 1, seed=run_seed, accuracy=acc, precision=prec,
                   recall=rec, f1=f1, auc=auc)
        run_rows.append(row)
        print(f"Run {run_idx + 1}/{n_runs}: accuracy={acc:.4f} precision={prec:.4f} "
              f"recall={rec:.4f} f1={f1:.4f} auc={auc:.4f}")
        logging.info(f"Run {run_idx + 1}/{n_runs} (seed={run_seed}): {row}")

        last_history = history
        last_model = model
        last_report = classification_report(
            y_test, y_pred, target_names=['benign', 'malignant'], zero_division=0
        )

    def mean_std(key):
        vals = [r[key] for r in run_rows]
        return float(np.mean(vals)), float(np.std(vals))

    acc_mean, acc_std = mean_std('accuracy')
    prec_mean, prec_std = mean_std('precision')
    rec_mean, rec_std = mean_std('recall')
    f1_mean, f1_std = mean_std('f1')
    auc_mean, auc_std = mean_std('auc')

    per_run_lines = "\n".join(
        f"  Run {r['run']} (seed={r['seed']}): accuracy={r['accuracy']:.4f} "
        f"precision={r['precision']:.4f} recall={r['recall']:.4f} "
        f"f1={r['f1']:.4f} auc={r['auc']:.4f}"
        for r in run_rows
    )

    summary = (
        f"Evaluation run: {start_time}\n"
        f"Split (random_state={random_state}): {len(f_train)} train / {len(f_val)} val / {len(f_test)} test\n"
        f"Test set files: {f_test}\n"
        f"Number of repeated training runs on this same split: {n_runs}\n\n"
        f"Per-run results:\n{per_run_lines}\n\n"
        f"Aggregate (mean +/- std over {n_runs} runs):\n"
        f"Accuracy:  {acc_mean:.4f} +/- {acc_std:.4f}\n"
        f"Precision: {prec_mean:.4f} +/- {prec_std:.4f}\n"
        f"Recall:    {rec_mean:.4f} +/- {rec_std:.4f}\n"
        f"F1-score:  {f1_mean:.4f} +/- {f1_std:.4f}\n"
        f"ROC-AUC:   {auc_mean:.4f} +/- {auc_std:.4f}\n\n"
        f"Summed confusion matrix across all {n_runs} runs "
        f"(rows=true, cols=predicted, order=[benign, malignant]):\n{summed_cm}\n\n"
        f"Classification report (last run only, run {n_runs}):\n{last_report}\n"
    )
    print(summary)
    logging.info(summary)

    metrics_path = os.path.join(output_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(summary)

    plot_training_curves(last_history, os.path.join(output_dir, "training_curves.png"))
    plot_confusion_matrix(summed_cm, os.path.join(output_dir, "confusion_matrix.png"))

    model_path = os.path.join(output_dir, f"eval_model_{start_time.strftime('%Y%m%d_%H%M%S')}.h5")
    last_model.save(model_path)

    print(f"Saved metrics to {metrics_path}")
    print(f"Saved training curves (last run) to {os.path.join(output_dir, 'training_curves.png')}")
    print(f"Saved summed confusion matrix plot to {os.path.join(output_dir, 'confusion_matrix.png')}")
    print(f"Saved last run's model to {model_path}")
    logging.info(f"Evaluation completed at {datetime.now()}. Total time: {datetime.now() - start_time}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train n_runs times on a stratified train/val split and evaluate on a "
                     "held-out test set, reporting mean +/- std across runs."
    )
    parser.add_argument("--input-dir", default="data/cbis_ddsm_processed", help="Directory of preprocessed .npy files")
    parser.add_argument("--label-csv", default="cbis_ddsm_selection.csv", help="CSV with folder_id,label columns")
    parser.add_argument("--output-dir", default="evaluation", help="Directory to write metrics.txt, plots, and the evaluated model")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs per run")
    parser.add_argument("--batch-size", type=int, default=2, help="Training batch size")
    parser.add_argument("--random-state", type=int, default=42,
                         help="Base random seed: used for the train/val/test split, and as the "
                              "starting seed for each run (run i uses seed random_state + i)")
    parser.add_argument("--n-runs", type=int, default=5,
                         help="Number of times to repeat training on the same split, to measure and report result variance")
    args = parser.parse_args()
    main(
        input_dir=args.input_dir,
        label_csv=args.label_csv,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        random_state=args.random_state,
        n_runs=args.n_runs,
    )
