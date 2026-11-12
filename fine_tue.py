import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset

from config import BOS_ID, EOS_ID, MAX_LEN, PAD_ID, UNK_ID
from data import tokenize
from transformer import Transformer


CHECKPOINT_PATH = "best_transformer.pt"
BASE_EXAMPLES = 10_000
EXTRA_EXAMPLES = 5_000
BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.00005


def load_finetuning_data(token_to_id):
    rows = load_dataset(
        "Helsinki-NLP/opus_books",
        "en-fr",
        split=f"train[:{BASE_EXAMPLES}]",
    )
    pairs = [
        (row["translation"]["en"], row["translation"]["fr"])
        for row in rows
        if row["translation"]["en"].strip()
        and row["translation"]["fr"].strip()
    ]

    # Recreate the original train/validation split.
    random.Random(42).shuffle(pairs)
    split_at = int(len(pairs) * 0.9)
    train_pairs = pairs[:split_at]
    val_pairs = pairs[split_at:]

    # Add rows after the original 10,000 to training only.
    extra_rows = load_dataset(
        "Helsinki-NLP/opus_books",
        "en-fr",
        split=f"train[{BASE_EXAMPLES}:{BASE_EXAMPLES + EXTRA_EXAMPLES}]",
    )
    train_pairs.extend(
        (row["translation"]["en"], row["translation"]["fr"])
        for row in extra_rows
        if row["translation"]["en"].strip()
        and row["translation"]["fr"].strip()
    )

    def encode_source(text):
        words = tokenize(text)[: MAX_LEN - 1]
        return [token_to_id.get(word, UNK_ID) for word in words] + [EOS_ID]

    def encode_target(text):
        words = tokenize(text)[: MAX_LEN - 2]
        return [BOS_ID] + [token_to_id.get(word, UNK_ID) for word in words] + [EOS_ID]

    encoded_train = [
        (encode_source(source), encode_target(target))
        for source, target in train_pairs
    ]
    encoded_val = [
        (encode_source(source), encode_target(target))
        for source, target in val_pairs
    ]

    def collate_batch(batch):
        sources, targets = zip(*batch)
        src = torch.full(
            (len(batch), max(len(seq) for seq in sources)),
            PAD_ID,
            dtype=torch.long,
        )
        tgt = torch.full(
            (len(batch), max(len(seq) for seq in targets)),
            PAD_ID,
            dtype=torch.long,
        )

        for i, (source, target) in enumerate(batch):
            src[i, : len(source)] = torch.tensor(source)
            tgt[i, : len(target)] = torch.tensor(target)

        return src, tgt

    train_loader = DataLoader(
        encoded_train,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        encoded_val,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_batch,
    )
    return train_loader, val_loader


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_tokens = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for src, tgt in loader:
            src = src.to(device)
            tgt = tgt.to(device)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            logits = model(src, tgt_input)
            loss = criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_output.reshape(-1),
            )

            if training:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            token_count = (tgt_output != PAD_ID).sum().item()
            total_loss += loss.item() * token_count
            total_tokens += token_count

    if total_tokens == 0:
        raise ValueError("The data loader produced no target tokens")
    return total_loss / total_tokens


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=True,
    )
    token_to_id = checkpoint["token_to_id"]
    id_to_token = checkpoint["id_to_token"]

    expected_special_ids = {
        "<pad>": PAD_ID,
        "<bos>": BOS_ID,
        "<eos>": EOS_ID,
        "<unk>": UNK_ID,
    }
    for token, expected_id in expected_special_ids.items():
        if token_to_id.get(token) != expected_id:
            raise ValueError(f"Checkpoint has an unexpected ID for {token}")

    train_loader, val_loader = load_finetuning_data(token_to_id)

    model = Transformer().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.98),
        eps=1e-9,
    )

    best_val_loss = run_epoch(model, val_loader, criterion, device)
    print(f"Starting validation loss: {best_val_loss:.4f}")

    for epoch in range(EPOCHS):
        train_loss = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss = run_epoch(model, val_loader, criterion, device)

        print(
            f"Epoch [{epoch + 1}/{EPOCHS}] "
            f"Train Loss: {train_loss:.4f} "
            f"Val Loss: {val_loss:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "token_to_id": token_to_id,
                    "id_to_token": id_to_token,
                },
                CHECKPOINT_PATH,
            )
            print("Saved improved checkpoint")


if __name__ == "__main__":
    main()