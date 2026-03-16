from Other_classes import (PatchEmbedding, 
TransformerEncoder, insert_mask_tokens, MAEDecoder, patchify)

from positional_encoding import Positional_Encoding
from torchvision import transforms
import torch
from torch import nn
from PIL import Image
import base64, io

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def random_masking(x, mask_ratio=0.75):
    
    """
    x: [B, N, D]
    """
    
    B, N, D = x.shape
    
    len_keep = int(N * (1 - mask_ratio))
    
    # ---- random noise generate ----
    noise = torch.rand(B, N, device=x.device)
    
    # ---- shuffle indices ----
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    
    # ---- keep first tokens ----
    ids_keep = ids_shuffle[:, :len_keep]
    
    x_masked = torch.gather(
        x,
        dim=1,
        index=ids_keep.unsqueeze(-1).repeat(1, 1, D)
    )
    
    # ---- create mask ----
    mask = torch.ones(B, N, device=x.device)
    mask[:, :len_keep] = 0
    
    # unshuffle mask
    mask = torch.gather(mask, dim=1, index=ids_restore)
    
    return x_masked, mask, ids_restore

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# patch_embed = PatchEmbedding().to(device)
# # batch = next(iter(dataloader))
# # print("batch_shape: ", batch.shape)

# img = Image.open("test_image.jpg").convert("RGB")
# img_tensor = transform(img).unsqueeze(0).to(device)  # [1,3,224,224]

# tokens = patch_embed(img_tensor)
# print("tokens_shape:" , tokens.shape)


# visible_tokens, mask, ids_restore = random_masking(tokens)


d_model = 768  # main model dimension
num_heads = 8  # number of heads
d_ff = 2048    # feedforward hidden dimension
seq_len = 196  # max input length
vocab_size = 30000

pos_encoding = Positional_Encoding(seq_len, d_model).to(device)
# visible_tokens = pos_encoding(visible_tokens)

encoder = TransformerEncoder(num_layers=8, d_model=768, d_ff=2048, num_heads=8).to(device)

# latent = encoder(visible_tokens)
# print("latent.shape: ", latent.shape)

mask_token = nn.Parameter(torch.zeros(1, 1, 768)).to(device)

# decoder = insert_mask_tokens(latent, ids_restore, mask_token)
# print("decoder_input_shape: ", decoder.shape)


class MAE_inference(nn.Module):

    def __init__(self, seq_len=196, embed_dim=768):
        super().__init__()

        self.patch_embed = PatchEmbedding()
        self.encoder = TransformerEncoder(
            num_layers=8,
            d_model=768,
            d_ff=2048,
            num_heads=8
        )

        self.decoder = MAEDecoder(   # ✅ MODULE
            depth=4,
            embed_dim=768,
            d_ff=2048,
            num_heads=8,
            patch_dim=768
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_encoding = Positional_Encoding(seq_len, embed_dim)

    def forward(self, images):

        tokens = self.patch_embed(images)
        visible_tokens, mask, ids_restore = random_masking(tokens)

        visible_tokens = self.pos_encoding(visible_tokens)
        latent = self.encoder(visible_tokens)

        decoder_input = insert_mask_tokens(
            latent, ids_restore, self.mask_token
        )

        pred = self.decoder(decoder_input)
        target = patchify(images)

        target = (target - target.mean(dim=-1, keepdim=True)) / \
                 (target.std(dim=-1, keepdim=True) + 1e-6)

        return pred, target, mask
    


def unpatchify(patches, patch_size=16, img_size=224):
    """
    patches: [B, N, C*ps*ps]
    return:  [B, C, H, W]
    """
    B, N, D = patches.shape
    C = 3
    h = w = img_size // patch_size  # 14

    patches = patches.view(B, N, C, patch_size, patch_size)
    # [B, N, C, ps, ps]

    patches = patches.view(B, h, w, C, patch_size, patch_size)
    # [B, 14, 14, C, ps, ps]

    patches = patches.permute(0, 3, 1, 4, 2, 5)
    # [B, C, 14, ps, 14, ps]

    images = patches.reshape(B, C, img_size, img_size)
    return images


checkpoint = torch.load("mae_epoch_new_69.pth", map_location=device)

model = MAE_inference().to(device)
model.load_state_dict(checkpoint["model"])  # 🔥 MAIN FIX
model.eval()


def tensor_to_base64(tensor):
    """
    tensor: [1,3,H,W]
    """
    img = tensor[0].cpu().permute(1,2,0)

    img = (img - img.min()) / (img.max() - img.min())
    img = (img * 255).byte().numpy()

    pil_img = Image.fromarray(img)

    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")

    img_str = base64.b64encode(buffer.getvalue()).decode()

    return img_str


# ---------- MAIN FUNCTION ----------
def run_mae_inference(image):

    img = Image.open(image).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred, target, mask = model(img_tensor)

    recon_img = unpatchify(pred)
    orig_img = img_tensor

    mask_img = mask.reshape(1, 14, 14)
    mask_img = mask_img.repeat_interleave(16, 1).repeat_interleave(16, 2)
    mask_img = mask_img.unsqueeze(1)

    final_img = orig_img * (1 - mask_img) + recon_img * mask_img

    return {
        "original": tensor_to_base64(orig_img),
        "reconstructed": tensor_to_base64(recon_img),
        "mae_output": tensor_to_base64(final_img)
    }