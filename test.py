import re
import torch
import random
from datasets import load_dataset
from config import BOS_ID,EOS_ID,MAX_LEN,UNK_ID
from data import tokenize
from transformer import Transformer

device="cuda" if torch.cuda.is_available() else "cpu"
checkpoint = torch.load(
    "best_transformer.pt",
    map_location=device,
    weights_only=True,
)
id_to_token = checkpoint["id_to_token"]
token_to_id = checkpoint["token_to_id"]

# Match the Transformer() setup used during training.
model=Transformer().to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

@torch.no_grad()
def translate(text):
    source_id=[
        token_to_id.get(token,UNK_ID)
        for token in tokenize(text)[:MAX_LEN-1]
    ]
    source_id.append(EOS_ID)
    src=torch.tensor([source_id],dtype=torch.long,device=device)
    # Start French generation with BOS.
    generated = torch.tensor([[BOS_ID]], dtype=torch.long, device=device)

    for _ in range(MAX_LEN-1):
        logits=model(src,generated)
        # The model output layer has VOCAB_SIZE outputs, but the savedHI
        # vocabulary may be smaller. Ignore unused output IDs.
        next_token_logits = logits[0, -1, :len(id_to_token)]
        next_id = next_token_logits.argmax().item()
        
        if next_id == EOS_ID:
            break
        generated=torch.cat([
            generated,
            torch.tensor([[next_id]],dtype=torch.long,device=device)
        ],
        dim=1,
        )
    words=[
        id_to_token[token_id] for token_id in generated[0,1:].tolist()
        if token_id not in (BOS_ID,EOS_ID)
    ]

    translation=" ".join(words)
    return re.sub(r"\s+([,.!?;:])", r"\1", translation)

# Use the same dataset size and split seed as training.
rows = load_dataset(
    "Helsinki-NLP/opus_books",
    "en-fr",
    split="train[:5000]",
)

pairs = [
    (row["translation"]["en"], row["translation"]["fr"])
    for row in rows
    if row["translation"]["en"].strip()
    and row["translation"]["fr"].strip()
]

random.Random(42).shuffle(pairs)
split_at = int(len(pairs) * 0.9)
validation_pairs = pairs[split_at:]

# Show model predictions beside the real translations.
for english, actual_french in validation_pairs[:5]:
    predicted_french = translate(english)

    print(f"\nEnglish:            {english}")
    print(f"Actual French:      {actual_french}")
    print(f"Model translation:  {predicted_french}")

