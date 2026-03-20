"""
Full pipeline orchestrator — scrape, process, retrain.

Usage:
    python run_pipeline_full.py           # scrape + process + retrain if new data
    python run_pipeline_full.py --no-scrape   # skip scraping, just process + retrain
    python run_pipeline_full.py --force-train  # force model retraining
"""
import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("orchestrator")


def main():
    parser = argparse.ArgumentParser(description="Nairobi AQ full pipeline")
    parser.add_argument("--no-scrape", action="store_true",
                        help="Skip the scraping step")
    parser.add_argument("--force-train", action="store_true",
                        help="Force Prophet model retraining")
    args = parser.parse_args()

    new_data = False

    # ── Step 1: Scrape ──
    if not args.no_scrape:
        from scraper import scrape_once
        log.info("=== STEP 1: Scraping new data ===")
        stats = scrape_once()
        new_data = stats.get("downloaded", 0) > 0 or stats.get("updated", 0) > 0
        if new_data:
            log.info("New/updated data detected.")
        else:
            log.info("No new data from scraper.")
    else:
        log.info("=== STEP 1: Scraping skipped ===")
        new_data = True  # assume we want to process

    # ── Step 2: Data pipeline ──
    if new_data or args.force_train:
        from src.pipeline import run_pipeline
        log.info("=== STEP 2: Running data pipeline ===")
        result = run_pipeline()
        log.info("Pipeline processed %d location(s).", len(result))
    else:
        log.info("=== STEP 2: Pipeline skipped (no new data) ===")

    # -- Step 3: Retrain models --
    if new_data or args.force_train:
        from src.processor import load_and_prepare_data
        from src.model import train_model, TRAIN_LOCATION

        log.info("=== STEP 3: Training Prophet models on location %d ===",
                 TRAIN_LOCATION)
        df = load_and_prepare_data(location=TRAIN_LOCATION)
        if df is not None and not df.empty:
            for target in ('P2', 'P1'):
                label = 'PM2.5' if target == 'P2' else 'PM10'
                model, metrics = train_model(df, force=True, target=target)
                log.info("%s model trained. MAE = %.2f", label, metrics['mae'])
        else:
            log.warning("No data for training location %d — skipping training.",
                        TRAIN_LOCATION)
    else:
        log.info("=== STEP 3: Training skipped (no new data) ===")

    log.info("=== Pipeline complete ===")


if __name__ == "__main__":
    main()
