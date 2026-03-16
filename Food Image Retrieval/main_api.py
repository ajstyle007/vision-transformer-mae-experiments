from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
import shutil
import os, glob, json, io, base64
import torch
import faiss
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware
from PIL import ImageDraw, ImageFont

from inference_vit import MAEEncoder
from inference_vit import get_embedding
from inference_vit import faiss_retrieve
from inference_vit import load_image_by_id


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Static & Templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Load model & FAISS once 
print("Loading model and FAISS index...")


ckpt_path = "mae_epoch_new_69.pth"

model = MAEEncoder().to(device)

ckpt = torch.load(ckpt_path, map_location=device)
state_dict = ckpt["model"]

model.load_state_dict(state_dict, strict=False)
model.eval()

@app.get("/")
def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})



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






index = faiss.read_index("mae_food.index")
image_names = np.load("image_ids.npy")


@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    temp_path = "temp.jpg"
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    query_emb = get_embedding(model, temp_path, device)
    results = faiss_retrieve(query_emb, index, image_names, top_k=6)

    os.remove(temp_path)

    images_base64 = []

    for item in results:
        image_id = item[0]
        score = item[1]
        
        # Load image from parquet
        img, label = load_image_by_id(image_id, parquet_files)
        if img is None:
            continue

        # # Draw label on image
        # draw = ImageDraw.Draw(img)
        # try:
        #     font = ImageFont.truetype("arial.ttf", 20)  # system font
        # except:
        #     font = ImageFont.load_default()
        
        # text = f"{label}\nSim: {score:.3f}"
        # # Draw a rectangle behind text for better visibility
        # # Get bounding box of the text
        # bbox = draw.textbbox((0,0), text, font=font)
        # text_width = bbox[2] - bbox[0]
        # text_height = bbox[3] - bbox[1]

        # # Draw rectangle behind text
        # draw.rectangle([0, 0, text_width + 10, text_height + 10], fill=(0, 0, 0, 128))

        # # Draw text
        # draw.text((5,5), text, fill="white", font=font)

    #     # Convert to base64
    #     buffered = io.BytesIO()
    #     img.save(buffered, format="JPEG")
    #     img_str = base64.b64encode(buffered.getvalue()).decode()
    #     images_base64.append(f"data:image/jpeg;base64,{img_str}")
    
    # return {"results": images_base64}

        results_list = []

        for item in results:
            image_id = item[0]
            score = float(item[1])

            img, label = load_image_by_id(image_id, parquet_files)
            if img is None:
                continue

            # convert image to base64
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()

            results_list.append({
                "image": f"data:image/jpeg;base64,{img_str}",
                "label": label,
                "score": round(score, 4)
            })

        return {"results": results_list}