from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from datasets import load_dataset

dataset = load_dataset("maywell/korean_textbooks", "tiny-textbooks", split="train")
texts = [item["text"] for item in dataset]

tokenizer = Tokenizer(models.WordPiece(unk_token="[UNK]"))
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

trainer = trainers.WordPieceTrainer(
    vocab_size=30522,
    min_frequency=2,
    special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"],
)

tokenizer.train_from_iterator(texts, trainer=trainer)
tokenizer.save("my_tokenizer.json")

# test tokenizer
text = "지금은 응용자연어처리 수업 중입니다."
encoded_t = tokenizer.encode(text)

# load tokenizer from file and test it
tokenizer_from_file = Tokenizer.from_file("my_tokenizer.json")

encoded_f = tokenizer_from_file.encode(text)


print("tokenized right after training:", encoded_t.tokens)
print("tokenized from file:           ", encoded_f.tokens)

print("ids right after tokenizing:    ", encoded_t.ids)
print("ids from file:                 ", encoded_f.ids)

"""
[00:00:08] Pre-processing sequences       ██████████████████████████████████████████████████████████████████████████████████████████████████████████████████ 0        /        0
[00:00:02] Tokenize words                 ██████████████████████████████████████████████████████████████████████████████████████████████████████████████████ 1564870  /  1564870
[00:00:01] Count pairs                    ██████████████████████████████████████████████████████████████████████████████████████████████████████████████████ 1564870  /  1564870
[00:00:07] Compute merges                 ██████████████████████████████████████████████████████████████████████████████████████████████████████████████████ 17958    /    17958
tokenized right after training: ['지금은', '응용', '##자연', '##어', '##처리', '수업', '중입니다', '.']
tokenized from file:            ['지금은', '응용', '##자연', '##어', '##처리', '수업', '중입니다', '.']
ids right after tokenizing:     [20583, 13379, 16249, 6751, 19769, 15561, 26182, 18]
ids from file:                  [20583, 13379, 16249, 6751, 19769, 15561, 26182, 18]
"""