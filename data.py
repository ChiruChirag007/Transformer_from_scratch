import re
import random
from collections import Counter
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from config import PAD_ID, BOS_ID, EOS_ID, UNK_ID, VOCAB_SIZE, MAX_LEN

def tokenize(text):
    return re.findall(r"\w+|[^\w\s]", text.lower())

def make_translation_loader(batch_size=32,num_examples=2000):
    rows = load_dataset(
        "Helsinki-NLP/opus_books",
        "en-fr",
        split=f"train[:{num_examples}]",
    )
    pairs = [
        (row["translation"]["en"], row["translation"]["fr"])
        for row in rows
        if row["translation"]["en"].strip() and row["translation"]["fr"].strip()
    ]

    random.Random(42).shuffle(pairs)
    split_at = int(len(pairs) * 0.9)
    train_pairs = pairs[:split_at]
    val_pairs = pairs[split_at:]

    # Build one shared vocabulary from both languages.
    counter = Counter()
    for source, target in train_pairs:
        counter.update(tokenize(source))
        counter.update(tokenize(target))
    special_tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]
    tokens = special_tokens + [
        token for token, _ in counter.most_common(VOCAB_SIZE - len(special_tokens))
        if token not in special_tokens
        ]
    token_to_id = {token: idx for idx, token in enumerate(tokens)}

    def encode_source(text):
        words = tokenize(text)[:MAX_LEN - 1]
        ids = [token_to_id.get(word, UNK_ID) for word in words]
        return ids + [EOS_ID]

    def encode_traget(text):
        words = tokenize(text)[:MAX_LEN - 2]
        ids = [token_to_id.get(word, UNK_ID) for word in words]
        return [BOS_ID] + ids + [EOS_ID]

    encoded_train = [
        (encode_source(source),encode_traget(target))
        for source,target in train_pairs
    ]
    encoded_val=[
        (encode_source(source),encode_traget(target))
        for source,target in val_pairs
    ]

    def collate_batch(batch):
        sources,targets = zip(*batch)

        src=torch.full(
            (len(batch),max(len(seq) for seq in sources)),
            PAD_ID,
            dtype=torch.long,
        )

        tgt=torch.full(
            (len(batch),max(len(seq) for seq in targets)),
            PAD_ID,
            dtype=torch.long,
        )

        for i,(source,target) in enumerate(batch):
            src[i, :len(source)] = torch.tensor(source)
            tgt[i, :len(target)] = torch.tensor(target)

        return src,tgt

    train_loader = DataLoader(
        encoded_train,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
    encoded_val,
    batch_size=batch_size,
    shuffle=False,
    collate_fn=collate_batch,
)

    return train_loader,val_loader,len(tokens),token_to_id,tokens