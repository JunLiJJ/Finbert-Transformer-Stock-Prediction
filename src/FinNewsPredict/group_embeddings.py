import argparse
import json
import random
from collections import defaultdict
from tqdm import tqdm
import os

"""
使用方法：
python group_embeddings.py \
  --input data/news_embeddings.jsonl \
  --output data/daily_embeddings.jsonl \
  --max_per_day 20
"""


def load_embeddings(jsonl_path):
    date_to_embeddings = defaultdict(list)
    with open(jsonl_path, "r") as f:
        for line in tqdm(f, desc="Loading embeddings"):
            item = json.loads(line)
            date = item["date"]
            emb = item["embedding"]
            date_to_embeddings[date].append(emb)
    return date_to_embeddings


def group_by_date(date_to_embeddings, max_per_day=None, sort=False):
    grouped = []
    for date, emb_list in date_to_embeddings.items():
        orig_len = len(emb_list)

        # Optional: sort or sample
        if max_per_day is not None and orig_len > max_per_day:
            if sort:
                emb_list = sorted(emb_list, key=lambda x: sum(x), reverse=True)[:max_per_day]
            else:
                emb_list = random.sample(emb_list, max_per_day)

        grouped.append({
            "date": date,
            "embedding_seq": emb_list,
            "length": orig_len
        })
    return grouped


def save_grouped(grouped_data, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        for item in grouped_data:
            f.write(json.dumps(item) + "\n")
    print(f"✅ Saved grouped data to {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Group per-news embeddings by date.")
    parser.add_argument("--input", type=str, required=True, help="Path to news_embeddings.jsonl")
    parser.add_argument("--output", type=str, required=True, help="Output file path for daily grouped embeddings")
    parser.add_argument("--max_per_day", type=str, default=None, help="Max number of embeddings to sample per day")
    parser.add_argument("--sort", action="store_true", help="Use top embeddings by sum instead of random sample")
    args = parser.parse_args()
    args.max_per_day = int(args.max_per_day) if args.max_per_day else None
    return args

def main():
    args = parse_args()
    date_to_embeddings = load_embeddings(args.input)
    grouped = group_by_date(date_to_embeddings, max_per_day=args.max_per_day, sort=args.sort)
    save_grouped(grouped, args.output)


if __name__ == "__main__":
    main()
