# run_analysis.py
import argparse, logging, os
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s - %(message)s")

def step_0_validate_raw():
    raw = Path("data/raw/news_data.csv")
    assert raw.exists(), "Missing data/raw/news_data.csv"
    logging.info("Raw data found: %s", raw)

def step_1_clean():
    # TODO: 读取 raw，做清洗，存到 data/processed/
    os.makedirs("data/processed", exist_ok=True)
    logging.info("Data cleaned -> data/processed/")

def step_2_train():
    # TODO: 训练/微调，产出模型到 artifacts/models/
    os.makedirs("artifacts/models", exist_ok=True)
    logging.info("Model trained -> artifacts/models/")

def step_3_report():
    # TODO: 生成 figures/tables/report
    os.makedirs("results/figures", exist_ok=True)
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/report", exist_ok=True)
    logging.info("Results generated -> results/")

def main(mode: str):
    step_0_validate_raw()
    step_1_clean()
    if mode != "light":
        step_2_train()
    step_3_report()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["full","light"], default="light",
                        help="light: 仅复现结果（不重训）；full: 端到端含训练")
    args = parser.parse_args()
    main(args.mode)
