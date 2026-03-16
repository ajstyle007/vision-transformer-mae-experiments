import torch
import torch.nn as nn
from enc_deco_blocks import MAEEncoder, Decoder, embedding_layer, pos_encoding, prepare_decoder_input, generate_subsequent_mask
from custom_tokenizer import tokenizer  # your trained tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# 1. Load encoder and decoder
# -------------------------
encoder = MAEEncoder().to(device)
encoder.eval()

decoder = Decoder(num_layers=6, d_model=768, d_ff=2048, num_heads=8).to(device)
decoder.eval()

# -------------------------
# 2. Dummy image input
# -------------------------
# replace with real image if needed
images = torch.randn(1, 3, 224, 224).to(device)

print("images shape: ", images.shape)

# -------------------------
# 3. Encode the image
# -------------------------
enc_out = encoder(images)          # [B, embed_dim] -> normalize?
# expand to sequence for cross-attention: [B, S_enc, D]
# enc_out = enc_out.unsqueeze(1)     # minimal: 1 token for testing

print("enc shape: ", enc_out)

# -------------------------
# 4. Start decoder input with [BOS] token
# -------------------------
bos_id = tokenizer.token_to_id("[BOS]")
dec_input_ids = [bos_id]
dec_inp = prepare_decoder_input(dec_input_ids).to(device)  # [1, 1, 768]
print("dec_inp: ", dec_inp.shape)

# -------------------------
# 5. Generate one next token
# -------------------------
mask = generate_subsequent_mask(dec_inp.size(1)).to(device)
with torch.no_grad():
    out = decoder(dec_inp, enc_out, mask)           # [1, seq_len, d_model]
    # project to vocab size
    vocab_size = tokenizer.get_vocab_size()
    logits = nn.Linear(768, vocab_size).to(device)(out)  # [1, 1, vocab_size]
    probs = torch.softmax(logits[:, -1, :], dim=-1)      # take last token
    next_token_id = torch.argmax(probs, dim=-1).item()

# -------------------------
# 6. Decode token to word
# -------------------------
next_word = tokenizer.id_to_token(next_token_id)

print("Next token ID:", next_token_id)
print("Next word prediction:", next_word)