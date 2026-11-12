import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformer import Transformer
from data import make_translation_loader
from config import BOS_ID, EOS_ID, PAD_ID, VOCAB_SIZE

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

#Create the Loader
train_loader, val_loader, vocab_size, token_to_id, id_to_token = make_translation_loader(
    batch_size=16,
    num_examples=10000,
)
print("Vocabulary size:", VOCAB_SIZE)

#Model initialization
model = Transformer()
model.to(device)

#Loss and optimizer
criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)
optimizer = torch.optim.Adam(
    model.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9
)

#Training Function
def train(model,dataloader):
    model.train()
    total_loss=0
    total_tokens=0
    for scr, tgt in dataloader:
        scr, tgt = scr.to(device), tgt.to(device)
        # Teacher forcing
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        logits = model(scr, tgt_input)
        loss = criterion(logits.reshape(-1, logits.size(-1)), tgt_output.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
         # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 1.0
        )
        optimizer.step()
        token_count = (tgt_output != PAD_ID).sum().item()
        total_loss += loss.item() * token_count
        total_tokens += token_count

    return total_loss / total_tokens

#Validation Function
@torch.no_grad()
def evaluate(model, dataloader):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for src, tgt in dataloader:
        src, tgt = src.to(device), tgt.to(device)

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        logits = model(src, tgt_input)

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            tgt_output.reshape(-1),
        )

        token_count = (tgt_output != PAD_ID).sum().item()
        total_loss += loss.item() * token_count
        total_tokens += token_count

    return total_loss / total_tokens

# Training
epochs=30
best_val_loss=float("inf")
for epoch in range(epochs):
    train_loss = train(model, train_loader)
    val_loss = evaluate(model, val_loader)

    print(
        f"Epoch [{epoch + 1}/{epochs}] "
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
            "best_transformer.pt",
        )
        print("Saved")