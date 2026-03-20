"""
Train the Prophet model on location 3966 (Kibera) data.

Usage:
    python train_model.py            # only trains if no saved model exists
    python train_model.py --force    # re-train from scratch
    python train_model.py --pipeline # run data pipeline first, then train
"""
import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger("train_model")


def main():
    parser = argparse.ArgumentParser(description="Train the Prophet models (PM2.5 + PM10)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-training even if a saved model exists")
    parser.add_argument("--pipeline", action="store_true",
                        help="Run the data pipeline before training")
    args = parser.parse_args()

    # Optional: run pipeline first
    if args.pipeline:
        from src.pipeline import run_pipeline
        log.info("Running data pipeline ...")
        result = run_pipeline()
        log.info("Pipeline processed %d location(s).", len(result))

    # Load data for training location (3966 = Kibera)
    from src.processor import load_and_prepare_data
    from src.model import train_model, load_or_train_model, TRAIN_LOCATION

    log.info("Loading data for location %d ...", TRAIN_LOCATION)
    df = load_and_prepare_data(location=TRAIN_LOCATION)

    if df is None or df.empty:
        log.error("No data available for location %d. "
                  "Run the pipeline first: python train_model.py --pipeline", TRAIN_LOCATION)
        sys.exit(1)

    log.info("Data loaded: %d rows, %s to %s",
             len(df), df['timestamp'].min(), df['timestamp'].max())

    for target in ('P2', 'P1'):
        label = 'PM2.5' if target == 'P2' else 'PM10'
        log.info("--- Training %s model ---", label)
        if args.force:
            model, metrics = train_model(df, force=True, target=target)
        else:
            model, metrics = load_or_train_model(df, target=target)
        log.info("%s model ready. MAE = %.2f, residual_std = %.2f",
                 label, metrics['mae'], metrics['residual_std'])


if __name__ == "__main__":
    main()