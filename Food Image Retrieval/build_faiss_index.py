import torch
import faiss
import pyarrow.parquet as pq
import numpy as np
import glob
import io
from PIL import Image
from torchvision import transforms

from inference_vit import MAEEncoder

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def bytes_to_tensor(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return transform(img)


model = MAEEncoder().to(device)
ckpt = torch.load("mae_epoch_new_69.pth", map_location=device)
model.load_state_dict(ckpt["model"], strict=False) 
model.eval()

dim = 768
# res = faiss.StandardGpuResources()
cpu_index = faiss.IndexFlatIP(dim)   # cosine similarity
# gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)  #0 = GPU id (single GPU)
image_ids = []


import pyarrow.parquet as pq

table = pq.read_table("../data/train-00000-of-00008-26f523e9bdcc2b9a.parquet")
print(table.schema)


# parquet_files = glob.glob("data/*.parquet")
parquet_files = glob.glob("../data/*.parquet")
batch_size = 64

@torch.no_grad()
def embed_batch(images):
    images = images.to(device)
    emb = model(images)
    emb = torch.nn.functional.normalize(emb, dim=-1)
    return emb.cpu().numpy().astype("float32")


for pq_file in parquet_files:
    print(f"Processing {pq_file}")
    table = pq.read_table(pq_file)
    df = table.to_pandas()

    images, ids = [], []

    for _, row in df.iterrows():
        img_bytes = row["image"]["bytes"]
        images.append(bytes_to_tensor(img_bytes))
        ids.append(row["id"])

        if len(images) == batch_size:
            images = torch.stack(images)
            emb = embed_batch(images)

            cpu_index.add(emb)
            # gpu_index.add(emb)
            image_ids.extend(ids)

            images, ids = [], []

    # last batch
    if len(images) > 0:
        images = torch.stack(images)
        emb = embed_batch(images)

        cpu_index.add(emb)
        image_ids.extend(ids)


# cpu_index = faiss.index_gpu_to_cpu(gpu_index)
faiss.write_index(cpu_index, "mae_food.index")
np.save("image_ids.npy", np.array(image_ids))

print("✅ FAISS index saved")
print("Total images indexed:", cpu_index.ntotal)

