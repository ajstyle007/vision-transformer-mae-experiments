from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("food_tokenizer.json")

encoded = tokenizer.encode("pizza with cheese color")

print(encoded.ids)
print(encoded.tokens)