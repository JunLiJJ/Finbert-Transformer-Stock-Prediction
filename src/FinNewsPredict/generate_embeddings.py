import os
import json
import torch
import argparse
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader

def parse_args():
    parser = argparse.ArgumentParser(description="Generate FinBert Embeddings")
    parser.add_argument("--dataset_name", type=str, default="sabareesh88/FNSPID_nasdaq")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--model_name", type=str, default="ProsusAI/finbert")
    parser.add_argument("--batch_size", type=int, default=32)
    # parser.add_argument("--output_file", type=str, default="news_embeddings.jsonl")
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=None, help="Optional limit on number of samples")
    args = parser.parse_args()
    
    # 自动命名输出文件
    if args.output_file is None:
        args.output_file = f"{args.dataset_name.replace('/', '_')}_embeddings_{args.model_name.replace('/', '_')}.jsonl"
    return args

# 自定义文本清洗函数
def clean_text(example):
    text = example["text"]
    if not text or len(text.strip()) < 10:
        return False  # 删除样本
    example["text"] = text.replace("\n", " ").strip()
    return example

def prepare_dataset(dataset, max_samples=None):
    print("🔧 Preprocessing dataset...")
    # # ✅ 先过滤掉 Luhn_summary 为空或空格的样本
    # dataset = dataset.filter(lambda x: x["Luhn_summary"] and x["Luhn_summary"].strip() != "")
    # 构建 text 字段，并保留 text + Date
    dataset = dataset.map(
        lambda x: {"text": x["Article_title"] + " " + x["Luhn_summary"]},
        remove_columns=[col for col in dataset.column_names if col not in ["text", "Date"]]
    )
    dataset = dataset.filter(lambda x: clean_text is not False)
    if max_samples:
        dataset = dataset.select(range(min(len(dataset), max_samples)))
    return dataset

def collate_fn(batch, tokenizer):
    texts = [example["text"] for example in batch]
    dates = [example["Date"] for example in batch]
    tokenized = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    tokenized["Date"] = dates
    return tokenized

def generate_embeddings(model, tokenizer, dataset, batch_size, device, output_file, flush_every=10000):
    print("🚀 Generating embeddings...")

    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=lambda x: collate_fn(x, tokenizer))
    model.to(device)
    model.eval()

    buffer = []

    with open(output_file, "w") as f_out, torch.no_grad(), autocast():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            dates = batch.pop("Date")
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu()  # [CLS] token

            for date, emb in zip(dates, cls_embeddings):
                buffer.append({
                    "date": date,
                    "embedding": emb.numpy().tolist()
                })

            # Flush buffer every N records
            if len(buffer) >= flush_every:
                f_out.write("\n".join(json.dumps(x) for x in buffer) + "\n")
                buffer = []

        # Flush remaining
        if buffer:
            f_out.write("\n".join(json.dumps(x) for x in buffer) + "\n")

    print(f"✅ Embeddings saved to {output_file}")

def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"📦 Loading dataset: {args.dataset_name} [{args.split}]")
    dataset = load_dataset(args.dataset_name, split=args.split, )

    dataset = prepare_dataset(dataset, max_samples=args.max_samples)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name)

    generate_embeddings(model, tokenizer, dataset, args.batch_size, device, args.output_file)

if __name__ == "__main__":
    main()

