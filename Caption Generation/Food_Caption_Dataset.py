import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import io
import pandas as pd
from torchvision import transforms
from tokenizers import Tokenizer
import pyarrow.parquet as pq
import pyarrow as pa
import pyarrow.dataset as ds

class FoodCaptionDataset(Dataset):
    def __init__(self, captions_path:str, images_path:str, tokenizer_path:str = "food_tokenizer.json", max_length:int=32):

        super().__init__()

        self.captions_df = pd.read_parquet(captions_path)

        pf = pq.ParquetFile(images_path)

        image_dict = {}
        for batch in pf.iter_batches(batch_size=1000):
            df = batch.to_pandas()

            for _, row in df.iterrows():
                image_dict[row["image_id"]] = row["image"]["bytes"]

        self.images_df = image_dict


        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_length = max_length

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


    def __len__(self):
        return len(self.captions_df)
    
    def __getitem__(self, idx):
        row = self.captions_df.iloc[idx]
        image_id = row["image_id"]
        caption  = row["caption"]

        # LAZY: Read ONLY this image from Parquet file
        img_bytes = self.images_df[image_id]
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        img_tensor = self.transform(img)

        # ── Tokenize caption ──
        caption = str(caption)
        encoding = self.tokenizer.encode(caption) # we already gets [BOS] ... [EOS]
        token_ids = encoding.ids # list[int]
        
        # For training we need input + target (shifted)
        input_ids  = token_ids[:-1]   # remove last [EOS] for input
        target_ids = token_ids[1:]    # remove first [BOS] for target

        # Pad / truncate
        input_ids_padded  = self._pad(input_ids)
        target_ids_padded = self._pad(target_ids)

        attention_mask = (torch.tensor(input_ids_padded) != self.tokenizer.token_to_id("[PAD]")).long()

        return {
            "image":      img_tensor,                # (3, 224, 224)
            "input_ids":  torch.tensor(input_ids_padded,  dtype=torch.long),
            "target_ids": torch.tensor(target_ids_padded, dtype=torch.long),
            "attention_mask": attention_mask,
            "length":     len(token_ids) - 1,        # useful for logging / masking loss
        }
    
    def _pad(self, token_list: list[int]) -> list[int]:
        if len(token_list) >= self.max_length:
            return token_list[:self.max_length]
        return token_list + [self.tokenizer.token_to_id("[PAD]")] * (self.max_length - len(token_list))
    


if __name__ == "__main__":

    dataset = FoodCaptionDataset(captions_path = "captions_new.parquet", images_path   = "images_new.parquet",
        max_length = 32,
    )

    # Quick test
    # sample = dataset[0]
    # print("Image shape:", sample["image"].shape)
    # print("Input ids length:", len(sample["input_ids"]))
    # print("Sample input:", sample["input_ids"][:15])
    # print("Sample target:", sample["target_ids"][:15])

    loader = DataLoader(dataset, batch_size=16, shuffle=True, drop_last=True)

    batch = next(iter(loader))
    print("\nBatch shapes:")
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            print(f"{k:15} → {v.shape}")



