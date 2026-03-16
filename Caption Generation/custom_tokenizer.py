import pandas as pd
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

captions_df = pd.read_parquet("captions_new.parquet")

captions = captions_df["caption"].tolist()

print(len(captions))

with open("captions.txt", "w", encoding="utf-8") as f:
    for c in captions:
        if c:
            f.write(str(c).strip() + "\n")

tokenizer = Tokenizer(BPE())

tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    vocab_size=8000,
    special_tokens=[
        "[PAD]",
        "[UNK]",
        "[BOS]",
        "[EOS]"
    ]
)

tokenizer.train(
    files=["captions.txt"],
    trainer=trainer
)


from tokenizers.processors import TemplateProcessing

tokenizer.post_processor = TemplateProcessing(
    single="[BOS] $A [EOS]",
    special_tokens=[
        ("[BOS]", tokenizer.token_to_id("[BOS]")),
        ("[EOS]", tokenizer.token_to_id("[EOS]")),
    ],
)


# SAVE TOKENIZER
tokenizer.save("food_tokenizer.json")

print("Tokenizer saved successfully")