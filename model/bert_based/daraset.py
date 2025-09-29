# dataset_bm25_minipack.py
import jsonlines, torch
from torch.utils.data import Dataset
from typing import List, Dict, Any
from pack_one_example_demo import pack_query_and_supports

def read_jsonl(p):
    rows = []
    with jsonlines.open(p, 'r') as r:
        for x in r: rows.append(x)
    return rows

def collect_all_labels(files: List[str]) -> List[str]:
    labs = set(["O"])
    for p in files:
        for r in read_jsonl(p):
            for t in r.get("tags", []): labs.add(t)
            for stags in r.get("supports_tags", []) or []:
                if isinstance(stags, list):
                    for t in stags: labs.add(t)
    return sorted(labs)

class BM25PackedNER(Dataset):
    def __init__(self, path: str, tokenizer, label_to_id: Dict[str,int], max_length: int = 512):
        self.rows = read_jsonl(path)
        self.tok = tokenizer
        self.l2i = label_to_id
        self.max_len = max_length

    def __len__(self): return len(self.rows)

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        r = self.rows[i]
        q_tokens = r.get("sentence") or r.get("tokens") or r["text"].split()
        q_tags   = r.get("tags")
        supports = r.get("supports", [])
        supports_tags = r.get("supports_tags", [])
        batch = pack_query_and_supports(
            tokenizer=self.tok,
            q_tokens=q_tokens,
            q_tags=q_tags,
            supports=supports,
            supports_tags=supports_tags,
            label_to_id=self.l2i,
            max_length=self.max_len,
            add_headers=True
        )
        # pack_query_and_supports returns 1-sample batch; unwrap to tensors
        return {
            "input_ids": batch["input_ids"].squeeze(0),
            "attention_mask": batch["attention_mask"].squeeze(0),
            "labels": batch["labels"].squeeze(0),
        }
