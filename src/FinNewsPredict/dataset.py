from torch.utils.data import Dataset
import torch
import random
import json

class DailyEmbeddingDataset(Dataset):
    """
    dataset = DailyEmbeddingDataset(
        jsonl_path="data/daily_embeddings.jsonl",
        label_dict=label_dict,
        max_seq_len=20
    )

    from torch.utils.data import DataLoader
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    """
    def __init__(self, jsonl_path, label_dict, max_seq_len=20, ignore_missing_label=True):
        self.max_seq_len = max_seq_len
        self.label_dict = label_dict
        self.ignore_missing_label = ignore_missing_label
        self.data = []

        with open(jsonl_path, "r") as f:
            for line in f:
                item = json.loads(line)
                date = item["date"]
                if date not in label_dict and ignore_missing_label:
                    continue
                self.data.append(item)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        embeddings = entry["embedding_seq"]
        orig_len = len(embeddings)

        if orig_len >= self.max_seq_len:
            sampled = random.sample(embeddings, self.max_seq_len)
        else:
            sampled = embeddings + [[0.0] * len(embeddings[0])] * (self.max_seq_len - orig_len)

        embedding_tensor = torch.tensor(sampled, dtype=torch.float32)
        attention_mask = torch.tensor(
            [1] * min(orig_len, self.max_seq_len) + [0] * max(0, self.max_seq_len - orig_len),
            dtype=torch.long
        )

        label = self.label_dict.get(entry["date"], -1)

        return {
            "date": entry["date"],
            "embedding_seq": embedding_tensor,  # (max_seq_len, dim)
            "attention_mask": attention_mask,
            "label": torch.tensor(label, dtype=torch.long)
        }
