import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import faiss
import numpy as np
import glob

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from Other_classes import PatchEmbedding, TransformerEncoder
from positional_encoding import Positional_Encoding

class MAEEncoder(nn.Module):
    def __init__(self, seq_len=196, embed_dim=768):
        super().__init__()

        self.patch_embed = PatchEmbedding()
        self.pos_encoding = Positional_Encoding(seq_len, embed_dim)

        self.encoder = TransformerEncoder(
            num_layers=8,
            d_model=embed_dim,
            d_ff=2048,
            num_heads=8
        )

    def forward(self, images):
        tokens = self.patch_embed(images)      # [B,196,768]
        tokens = self.pos_encoding(tokens)
        latent = self.encoder(tokens)          # [B,196,768]

        emb = latent.mean(dim=1)               # mean pooling
        emb = F.normalize(emb, dim=-1)

        return emb                             # [B,768]


index = faiss.read_index("mae_food.index")
image_names = np.load("image_ids.npy")


device = "cuda" if torch.cuda.is_available() else "cpu"

ckpt_path = "mae_epoch_new_69.pth"

model = MAEEncoder().to(device)

ckpt = torch.load(ckpt_path, map_location=device)
state_dict = ckpt["model"]

# 🔥 sirf encoder related weights load honge
model.load_state_dict(state_dict, strict=False)
model.eval()

print("✅ MAE Encoder loaded successfully")


transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

@torch.no_grad()
def get_embedding(model, image_path, device):
    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    embedding = model(img)   # 🔥 direct call
    return embedding.squeeze(0)


def faiss_retrieve(query_emb, index, image_ids, top_k=5):
    query_np = query_emb.cpu().numpy().astype("float32").reshape(1, -1)
    scores, indices = index.search(query_np, top_k)

    results = []
    for idx, score in zip(indices[0], scores[0]):
        results.append((image_ids[idx], float(score)))

    return results


query_img = "chawal.jpg"


import pyarrow.parquet as pq
from PIL import Image
import io, json


# ────────────────────────────────────────────────
#   Load label names ONCE from the first file
# ────────────────────────────────────────────────
label_names = None

parquet_files = glob.glob("../data/*.parquet")

if parquet_files:  # make sure we have at least one file
    try:
        # Option A: fastest — only read metadata + schema (no full data)
        # parquet_file = pq.ParquetFile(parquet_files[0])
        table = pq.read_table(parquet_files[0])
        meta = table.schema.metadata
        
        if meta and b'huggingface' in meta:
            hf_info = json.loads(meta[b'huggingface'].decode('utf-8'))
            label_names = hf_info.get("info", {}).get("features", {}).get("label", {}).get("names", None)
        
        # Fallback: if above fails, read small table (very rare)
        if label_names is None:
            table = pq.read_table(parquet_files[0], columns=["label"])  # only 1 column → fast
            meta = table.schema.to_arrow_schema().metadata
            # meta = parquet_file.schema.metadata
            hf_info = json.loads(meta[b'huggingface'].decode('utf-8'))
            label_names = hf_info["info"]["features"]["label"]["names"]

        print(f"Loaded {len(label_names)} class names from {parquet_files[0]}")
        print("Example:", label_names[:5])

    except Exception as e:
        print("Could not load label names from metadata:", str(e))
        label_names = []   # or raise error — your choice

else:
    print("No parquet files found → cannot load label names")


def load_image_by_id(image_id, parquet_files):
    """
    Search across parquet files using pyarrow only.
    Avoid full to_pandas() conversion.
    """
    target_id = image_id  # assuming image_id is int or str - make sure types match!

    for pq_path in parquet_files:
        try:
            # Option 1: Use dataset API (recommended for multiple files)
            # But if you want to keep simple loop:
            table = pq.read_table(
                pq_path,
                columns=["id", "image", "label"],          # only needed columns!
                filters=[("id", "=", target_id)]           # push down filter → huge speedup
            )
            
            if table.num_rows == 0:
                continue

            # Now table has at most 1 row
            row = table.slice(0, 1)  # safe even if >1 (shouldn't happen)

            img_bytes = row["image"][0]["bytes"].as_py()   # extract bytes
            label_idx = int(row["label"][0].as_py())       # convert to int

            img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            if label_names and 0 <= label_idx < len(label_names):
                label_str = label_names[label_idx]
            else:
                label_str = f"Class {label_idx} (name missing)"

            return img, label_str

        except Exception as e:
            print(f"Error reading {pq_path}: {e}")
            continue

    return None, "Not found"



import matplotlib.pyplot as plt

def visualize_results(query_image_path, results, image_cache):
    fig = plt.figure(figsize=(15, 4))

    # Query image
    query_img = Image.open(query_image_path).convert("RGB")
    ax = plt.subplot(1, len(results) + 1, 1)
    ax.imshow(query_img)
    ax.set_title("Query")
    ax.axis("off")

    # Retrieved images
    for i, (img_id, score) in enumerate(results):
        img, label = load_image_by_id(img_id, image_cache)

        ax = plt.subplot(1, len(results) + 1, i + 2)

        if img is None:
            ax.set_title(f"ID {img_id}\n❌ Not found")
            ax.axis("off")
            continue

        ax.imshow(img)
        ax.set_title(
            f"ID: {img_id}\nLabel: {label}\nSim: {score:.3f}",
            fontsize=9
        )
        ax.axis("off")

    plt.tight_layout()
    plt.show()




query_emb = get_embedding(model, query_img, device)

image_ids = np.load("image_ids.npy").tolist()
# retrieve
results = faiss_retrieve(query_emb, index, image_ids, top_k=5)

parquet_files = glob.glob("../data/*.parquet")

# build cache ONCE
# image_cache = build_image_cache(parquet_files)

# visualize
# visualize_results(query_img, results, parquet_files)