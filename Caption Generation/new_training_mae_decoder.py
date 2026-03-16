import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import faiss
import numpy as np
import glob
from Other_classes import Cross_Attention
from multihead_attention import MultiHeadAttention
from feed_forward_nn import feedforward
from tokenizers import Tokenizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from Other_classes import PatchEmbedding, TransformerEncoder
from positional_encoding import Positional_Encoding
from enc_deco_blocks import MAEEncoder, Decoder, generate_subsequent_mask, prepare_decoder_input
from Food_Caption_Dataset import FoodCaptionDataset

d_model = 768  # main model dimension
num_heads = 8  # number of heads
d_ff = 2048    # feedforward hidden dimension
seq_len = 32  # max input length
vocab_size = 30000

device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt_path = "mae_epoch_new_69.pth"

# model = MAEEncoder()
encoder = MAEEncoder().to(device)

ckpt = torch.load(ckpt_path, map_location=device)
state_dict = ckpt["model"]

encoder.load_state_dict(state_dict, strict=False)
for param in encoder.parameters():
    param.requires_grad = False

encoder.eval()

print("✅ MAE Encoder loaded successfully")


tokenizer = Tokenizer.from_file("food_tokenizer.json")
vocab_size = tokenizer.get_vocab_size()
lm_head = nn.Linear(768, vocab_size).to(device)


embedding_layer = nn.Embedding(vocab_size, d_model).to(device)

pos_encoding = Positional_Encoding(seq_len, d_model).to(device)

decoder = Decoder(num_layers=6, d_model=768, d_ff=2048, num_heads=8).to(device)


def generate_caption(image_path, max_len=32):

    encoder.eval()
    decoder.eval()

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        enc_out = encoder(image)

    bos = tokenizer.token_to_id("[BOS]")
    eos = tokenizer.token_to_id("[EOS]")

    caption = [bos]

    for _ in range(max_len):

        dec_inp = prepare_decoder_input(caption).to(device)

        mask = generate_subsequent_mask(dec_inp.size(1)).to(device)

        with torch.no_grad():
            out = decoder(dec_inp, enc_out, mask)
            logits = lm_head(out)

        temperature = 0.8
        probs = torch.softmax(logits[:, -1, :] / temperature, dim=-1)
        next_token = torch.multinomial(probs, 1).item()

        caption.append(next_token)

        if next_token == eos:
            break

    text = tokenizer.decode(caption)

    # print("\n🧾 Generated Caption:")
    print(text)

    encoder.train()
    decoder.train()


criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)  # ignore [PAD]

optimizer = torch.optim.AdamW(
    list(decoder.parameters()) +
    list(lm_head.parameters()) +
    list(embedding_layer.parameters()),
    lr=3e-4
)


dataset = FoodCaptionDataset(captions_path = "captions_new.parquet", images_path   = "images_new.parquet",
        max_length = 32,
    )

train_loader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=True)


os.makedirs("checkpoints", exist_ok=True)

resume_path = None   # put checkpoint path if resuming

start_epoch = 0
global_step = 0

resume_path = "checkpoints/decoder_step_280000.pth"

if resume_path is not None:
    ckpt = torch.load(resume_path, map_location=device)

    decoder.load_state_dict(ckpt["decoder"])
    lm_head.load_state_dict(ckpt["lm_head"])
    embedding_layer.load_state_dict(ckpt["embedding"])
    optimizer.load_state_dict(ckpt["optimizer"])

    start_epoch = ckpt["epoch"] 
    global_step = ckpt["step"]

    print(f"✅ Resumed from step {global_step}, epoch {start_epoch}")


log_file = open("training_log.txt", "a")



epochs = 7

for epoch in range(start_epoch, epochs):

    total_loss = 0
    

    # pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
    # resume progress bar correctly
    if epoch == start_epoch:
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}", initial=global_step % len(train_loader))
    else:
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")

    for batch in pbar:

        images = batch["image"].to(device)
        decoder_input = batch["input_ids"].to(device)
        target_tokens = batch["target_ids"].to(device)

        # ---------------- Encoder ----------------
        with torch.no_grad():
            enc_out = encoder(images)

        # ---------------- Decoder input ----------------
        dec_embed = embedding_layer(decoder_input)
        dec_embed = pos_encoding(dec_embed)

        mask = generate_subsequent_mask(dec_embed.size(1)).to(dec_embed.device)

        # ---------------- Decoder ----------------
        dec_out = decoder(dec_embed, enc_out, mask)

        logits = lm_head(dec_out)

        B,S,V = logits.shape
        logits = logits.view(B*S, V)
        targets = target_tokens.reshape(B*S)

        loss = criterion(logits, targets)

        # ---------------- Backprop ----------------
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(list(decoder.parameters()) + list(lm_head.parameters()) + list(embedding_layer.parameters()), 1.0)

        optimizer.step()

        global_step += 1


        total_loss += loss.item()

        pbar.set_postfix({"loss": loss.item(), "step": global_step})


        if global_step % 1000 == 0:
            log_msg = f"Epoch {epoch+1} Step {global_step} Loss {loss.item():.4f}"
            print(log_msg)

            log_file.write(log_msg + "\n")
            log_file.flush()


        if global_step % 10000 == 0:

            save_dict = {
                "step": global_step,
                "epoch": epoch,
                "decoder": decoder.state_dict(),
                "lm_head": lm_head.state_dict(),
                "embedding": embedding_layer.state_dict(),
                "optimizer": optimizer.state_dict()
            }

            save_path = f"checkpoints/decoder_step_{global_step}.pth"

            torch.save(save_dict, save_path)

            print(f"\n💾 Checkpoint saved: {save_path}")

            print("\n🔎 Generating sample caption...")
            generate_caption("test_food.jpg")


    epoch_ckpt = {
        "step": global_step,
        "epoch": epoch,
        "decoder": decoder.state_dict(),
        "lm_head": lm_head.state_dict(),
        "embedding" : embedding_layer.state_dict(),
        "optimizer": optimizer.state_dict()
    }

    torch.save(epoch_ckpt, f"checkpoints/decoder_epoch_{epoch}.pth")

    print(f"Epoch {epoch+1} Loss:", total_loss/len(train_loader))

log_file.close()
        

    