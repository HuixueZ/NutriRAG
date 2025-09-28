# build_topk_with_bm25.py
import argparse, json, jsonlines, re
from rank_bm25 import BM25Okapi
from typing import List, Dict

def read_jsonl(path):
    out = []
    with jsonlines.open(path, 'r') as r:
        for x in r:
            out.append(x)
    return out

def to_tokens(row) -> List[str]:
    # prefer 'tokens'; fallback to 'sentence'
    if "tokens" in row and isinstance(row["tokens"], list):
        return [str(t) for t in row["tokens"]]
    if "sentence" in row and isinstance(row["sentence"], list):
        return [str(t) for t in row["sentence"]]
    if "text" in row and isinstance(row["text"], str):
        # very light tokenization
        return re.findall(r"\w+|\S", row["text"])
    raise ValueError("Row must contain 'tokens', 'sentence', or 'text'.")

def pool_text(row) -> str:
    # human-readable sentence for writing out
    if "tokens" in row: return " ".join(map(str, row["tokens"]))
    if "sentence" in row: return " ".join(map(str, row["sentence"]))
    if "text" in row: return row["text"]
    return ""

def get_tags(row) -> List[str]:
    # optional; not all splits may have tags
    return row.get("tags", None)

def build_bm25(corpus_rows: List[Dict]):
    tokenized_corpus = [to_tokens(r) for r in corpus_rows]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, tokenized_corpus

def topk_from_train(query_tokens: List[str],
                    bm25: BM25Okapi,
                    k: int,
                    train_rows: List[Dict],
                    avoid_self_idx: int = None) -> List[int]:
    scores = bm25.get_scores(query_tokens)
    # mask self if requested
    if avoid_self_idx is not None and 0 <= avoid_self_idx < len(scores):
        scores[avoid_self_idx] = float("-inf")
    # partial sort then sort top k
    idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return idxs[:k]

def augment_split(split_rows: List[Dict],
                  train_rows: List[Dict],
                  bm25: BM25Okapi,
                  k: int,
                  is_train: bool) -> List[Dict]:
    augmented = []
    for i, row in enumerate(split_rows):
        q_tokens = to_tokens(row)
        avoid = i if is_train else None
        nbrs = topk_from_train(q_tokens, bm25, k, train_rows, avoid_self_idx=avoid)

        supports = [pool_text(train_rows[j]) for j in nbrs]
        supports_tags = []
        for j in nbrs:
            t = get_tags(train_rows[j])
            supports_tags.append(t if t is not None else None)

        new_row = dict(row)  # shallow copy
        new_row["supports"] = supports
        new_row["supports_tags"] = supports_tags
        augmented.append(new_row)
    return augmented

def write_jsonl(path, rows: List[Dict]):
    with jsonlines.open(path, 'w') as w:
        for r in rows:
            w.write(r)

def main():
    ap = argparse.ArgumentParser()
    #/Users/kz34/Documents/UMN/pycharm_project/nutrition/nutrition_llama_test/train.jsonl
    ap.add_argument("--indir", required=True, help="Folder with train.jsonl, valid.jsonl, test.jsonl")
    ap.add_argument("--outdir", required=True, help="Where augmented files will be written")
    ap.add_argument("--k", type=int, default=15, help="Top-k supports to retrieve from TRAIN")
    args = ap.parse_args()

    train = read_jsonl(f"{args.indir}/train.jsonl")
    valid = read_jsonl(f"{args.indir}/valid.jsonl")
    test  = read_jsonl(f"{args.indir}/test.jsonl")

    bm25, _ = build_bm25(train)

    train_aug = augment_split(train, train, bm25, args.k, is_train=True)
    valid_aug = augment_split(valid, train, bm25, args.k, is_train=False)
    test_aug  = augment_split(test,  train, bm25, args.k, is_train=False)

    write_jsonl(f"{args.outdir}/train.top{args.k}.jsonl", train_aug)
    write_jsonl(f"{args.outdir}/valid.top{args.k}.jsonl", valid_aug)
    write_jsonl(f"{args.outdir}/test.top{args.k}.jsonl",  test_aug)

if __name__ == "__main__":
    main()
