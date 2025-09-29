# train_with_minipack.py
import os, argparse, torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForTokenClassification
from tqdm.auto import tqdm
from seqeval.metrics import  classification_report,f1_score,recall_score, precision_score
from torch.optim import AdamW


from dataset_bm25_minipack import BM25PackedNER, collect_all_labels, read_jsonl
from pack_one_example_demo import IGNORE_INDEX, build_label_to_id
from math import isnan

def gather_preds(logits, labels):
    pred_ids = logits.argmax(-1)   # [B,T]
    B, T = labels.shape
    pred_list, true_list = [], []
    for b in range(B):
        p, y = [], []
        for t in range(T):
            if labels[b, t].item() == IGNORE_INDEX: 
                continue
            y.append(labels[b, t].item())
            p.append(pred_ids[b, t].item())
        pred_list.append(p); true_list.append(y)
    return pred_list, true_list

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True, help="dir containing train.topK.jsonl, valid.topK.jsonl, test.topK.jsonl")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--patience", default=3)
    args = ap.parse_args()

    train_p = os.path.join(args.indir, f"train.top{args.k}.jsonl")
    valid_p = os.path.join(args.indir, f"valid.top{args.k}.jsonl")
    test_p  = os.path.join(args.indir, f"test.top{args.k}.jsonl")

    # Label map from all splits
    labels = collect_all_labels([train_p, valid_p, test_p])
    l2i = build_label_to_id(labels)
    i2l = {v:k for k,v in l2i.items()}

    print("loading mdoel:")
    model_id="bert-base-uncased",
    model_id="bionlp/bluebert_pubmed_mimic_uncased_L-12_H-768_A-12"
    tok = BertTokenizer.from_pretrained(model_id, use_fast=False)
    model = BertForTokenClassification.from_pretrained(model_id, num_labels=len(l2i))

    print("processing mdoel:")

    tr_ds = BM25PackedNER(train_p, tok, l2i, max_length=args.max_len)
    va_ds = BM25PackedNER(valid_p, tok, l2i, max_length=args.max_len)
    te_ds = BM25PackedNER(test_p,  tok, l2i, max_length=args.max_len)

    tr_ld = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True)
    va_ld = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False)
    te_ld = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optim = AdamW(model.parameters(), lr=args.lr)

    print("training mdoel:")

    best_f1 = -1.0
    patience = args.patience
    epochs_since_improve = 0
    past_f1=[]
    for epoch in range(1, args.epochs+1):
        model.train()
        tot = 0.0
        for batch in tqdm(tr_ld, desc=f"Train {epoch}"):
            batch = {k:v.to(device) for k,v in batch.items()}
            out = model(**batch)
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step(); optim.zero_grad()
            tot += float(loss)
        print(f"Epoch {epoch}: train_loss={tot/len(tr_ld):.4f}")

        # eval
        model.eval()
        all_preds, all_trues = [], []
        with torch.no_grad():
            for batch in tqdm(va_ld, desc="Valid"):
                batch = {k:v.to(device) for k,v in batch.items()}
                out = model(**batch)
                preds, trues = gather_preds(out.logits, batch["labels"])
                # map ids -> tags
                pred_tags = [[i2l[i] for i in seq] for seq in preds]
                true_tags = [[i2l[i] for i in seq] for seq in trues]
                all_preds.extend(pred_tags); all_trues.extend(true_tags)
        micro = f1_score(all_trues, all_preds, average="micro")
        macro = f1_score(all_trues, all_preds, average="macro")
        print(f"Valid microF1={micro:.4f} macroF1={macro:.4f}")
        past_f1.append(micro)

    
        if micro > best_f1:
            best_f1 = micro
            epochs_since_improve = 0
            os.makedirs(args.outdir, exist_ok=True)
            save_path = os.path.join(args.outdir, f"top{args.k}_best_ep{epoch}_micro{micro:.4f}")
            model.save_pretrained(save_path)
            tok.save_pretrained(save_path)
            print(f"↑ New best micro-F1: {best_f1:.4f} (saved to {save_path})")
        elif best_f1!=-1.0:
            epochs_since_improve += 1
            print(f"No improvement ({epochs_since_improve}/{patience})")

        if epochs_since_improve >= patience:
            print(f"Early stopping: micro-F1 did not improve for {patience} consecutive epochs.")
            break

    # test
    print("\nTesting best/current weights:")
    model.eval()
    all_preds, all_trues = [], []
    with torch.no_grad():
        for batch in tqdm(te_ld, desc="Test"):
            batch = {k:v.to(device) for k,v in batch.items()}
            out = model(**batch)
            preds, trues = gather_preds(out.logits, batch["labels"])
            pred_tags = [[i2l[i] for i in seq] for seq in preds]
            true_tags = [[i2l[i] for i in seq] for seq in trues]
            all_preds.extend(pred_tags); all_trues.extend(true_tags)

    micro_precision=precision_score(all_trues, all_preds, average="micro")
    micro_recall=recall_score(all_trues, all_preds, average="micro")
    micro = f1_score(all_trues, all_preds, average="micro")

    weighted_precision=precision_score(all_trues, all_preds, average="weighted")
    weighted_recall=recall_score(all_trues, all_preds, average="weighted")
    weighted = f1_score(all_trues, all_preds, average="weighted")


    macro_precision=precision_score(all_trues, all_preds, average="macro")
    macro_recall=recall_score(all_trues, all_preds, average="macro")
    macro = f1_score(all_trues, all_preds, average="macro")


    print(f"TEST micro pre={micro_precision:.4f}")
    print(f"TEST mciro recall={micro_recall:.4f}")
    print(f"TEST mciro F1={micro:.4f}")

    print(f"TEST weighted pre={weighted_precision:.4f}")
    print(f"TEST weighted recall={weighted_recall:.4f}")
    print(f"TEST weighted F1={weighted:.4f}")

    print(f"TEST weighted pre={macro_precision:.4f}")
    print(f"TEST weighted recall={macro_recall:.4f}")
    print(f"TEST weighted F1={macro:.4f}")

    print(f"past f1: {past_f1}")



    #print(f"TEST microF1={micro:.4f} macroF1={macro:.4f}")
    print(classification_report(all_trues, all_preds))

if __name__ == "__main__":
    main()
