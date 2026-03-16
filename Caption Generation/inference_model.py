import torch
import torch.nn as nn
from enc_deco_blocks import MAEEncoder, Decoder, prepare_decoder_input, generate_subsequent_mask
from tokenizers import Tokenizer

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------
# Load encoder
# -------------------------
encoder = MAEEncoder().to(device)
encoder.eval()

# -------------------------
# Load decoder
# -------------------------
decoder = Decoder(num_layers=6, d_model=768, d_ff=2048, num_heads=8).to(device)

# LM head
tokenizer = Tokenizer.from_file("food_tokenizer.json")
vocab_size = tokenizer.get_vocab_size()
lm_head = nn.Linear(768, vocab_size).to(device)

# -------------------------
# Load checkpoint
# -------------------------
ckpt = torch.load("checkpoints/decoder_epoch_1.pth", map_location=device)

decoder.load_state_dict(ckpt["decoder"])
lm_head.load_state_dict(ckpt["lm_head"])

decoder.eval()
lm_head.eval()

print("✅ Checkpoint loaded")



from PIL import Image
from torchvision import transforms

image_path = "image.jpg"   # your image path

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

image = Image.open(image_path).convert("RGB")

image = transform(image)      # [3,224,224]

image = image.unsqueeze(0)    # [1,3,224,224]

image = image.to(device)

print(image.shape)


with torch.no_grad():
    enc_out = encoder(image)

print(enc_out.shape)


# enc_out = enc_out.unsqueeze(1)   # [B,1,768]

bos_id = tokenizer.token_to_id("[BOS]")
eos_id = tokenizer.token_to_id("[EOS]")

dec_input_ids = [bos_id]

max_len = 20

for step in range(max_len):

    dec_inp = prepare_decoder_input(dec_input_ids).to(device)

    mask = generate_subsequent_mask(dec_inp.size(1)).to(device)

    with torch.no_grad():
        out = decoder(dec_inp, enc_out, mask)

        logits = lm_head(out)

        next_token_logits = logits[:, -1, :]

        temperature = 0.5

        probs = torch.softmax(next_token_logits / temperature, dim=-1)

        next_token_id = torch.multinomial(probs, 1).item()

    dec_input_ids.append(next_token_id)

    if next_token_id == eos_id:
        break



tokens = [tokenizer.id_to_token(i) for i in dec_input_ids]

caption = " ".join(tokens)

print("Generated Caption:")
print(caption)


