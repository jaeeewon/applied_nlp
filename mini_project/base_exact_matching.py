import nltk, evaluate
from nltk.corpus import cmudict
from datasets import load_dataset

try:
    nltk.data.find("corpora/cmudict.zip")
except LookupError:
    nltk.download("cmudict")


class ExactMatchingP2G:
    def __init__(self):
        self.cmu_dict = cmudict.dict()
        self.inverted_dict = self._invert_dict()

    def _invert_dict(self):
        inv_dict = {}
        for word, prons in self.cmu_dict.items():
            for pron in prons:
                pron_tuple = tuple(pron)

                if pron_tuple not in inv_dict:
                    inv_dict[pron_tuple] = word
                # else:
                #     inv_dict[pron_tuple] = "<COL>"
        return inv_dict

    def decode(self, phoneme_list):
        decoded_words = []
        current_word_phonemes = []

        # additional " " for the last word
        processing_list = phoneme_list + [" "]

        for p in processing_list:
            if p == " ":
                if current_word_phonemes:
                    p_tuple = tuple(current_word_phonemes)

                    word = self.inverted_dict.get(p_tuple, "<UNK>")
                    decoded_words.append(word)

                    current_word_phonemes = []
            else:
                current_word_phonemes.append(p)

        return " ".join(decoded_words).upper()

    def decode_batch(self, batch):
        decoded_texts = []
        for phoneme_list in batch["phonemes"]:
            decoded_text = self.decode(phoneme_list)
            decoded_texts.append(decoded_text)
        return {"base_text": decoded_texts}


if __name__ == "__main__":
    baseline_model = ExactMatchingP2G()

    raw_datasets = load_dataset("jaeeewon/librispeech_phonemes")

    bleu_metric = evaluate.load("sacrebleu")
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    rouge_metric = evaluate.load("rouge")

    splits = [
        "dev.clean",
        "dev.other",
        "test.clean",
        "test.other",
    ]
    columns_to_remove = ["id", "text", "phonemes"]

    for split_name in splits:
        tokenized = raw_datasets[split_name].map(
            baseline_model.decode_batch,
            batched=True,
            remove_columns=columns_to_remove,
            desc=f"exact-matching {split_name}",
        )

        decoded_preds = tokenized["base_text"]
        decoded_labels = raw_datasets[split_name]["text"]

        bleu = bleu_metric.compute(
            predictions=decoded_preds,
            references=decoded_labels,
        )
        wer = wer_metric.compute(
            predictions=decoded_preds,
            references=decoded_labels,
        )
        cer = cer_metric.compute(
            predictions=decoded_preds,
            references=decoded_labels,
        )
        rouge = rouge_metric.compute(
            predictions=decoded_preds,
            references=decoded_labels,
            use_stemmer=True,
        )

        print(f"===== results for {split_name} =====")
        print(f"{split_name} BLEU   : {bleu['score']:.4f}")
        print(f"{split_name} WER    : {wer:.4f}")
        print(f"{split_name} CER    : {cer:.4f}")
        print(f"{split_name} ROUGE1 : {rouge['rouge1']:.4f}")
        print(f"{split_name} ROUGE2 : {rouge['rouge2']:.4f}")
        print(f"{split_name} ROUGEL : {rouge['rougeL']:.4f}")
        print()

"""
exact-matching dev.clean: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2703/2703 [00:00<00:00, 22814.30 examples/s]
===== results for dev.clean =====
dev.clean BLEU   : 47.3316
dev.clean WER    : 0.2562
dev.clean CER    : 0.1028
dev.clean ROUGE1 : 0.7456
dev.clean ROUGE2 : 0.5570
dev.clean ROUGEL : 0.7452

exact-matching dev.other: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2864/2864 [00:00<00:00, 25006.48 examples/s]
===== results for dev.other =====
dev.other BLEU   : 44.1307
dev.other WER    : 0.2788
dev.other CER    : 0.1161
dev.other ROUGE1 : 0.7171
dev.other ROUGE2 : 0.5222
dev.other ROUGEL : 0.7165

exact-matching test.clean: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2620/2620 [00:00<00:00, 21543.45 examples/s]
===== results for test.clean =====
test.clean BLEU   : 47.4273
test.clean WER    : 0.2579
test.clean CER    : 0.1029
test.clean ROUGE1 : 0.7448
test.clean ROUGE2 : 0.5614
test.clean ROUGEL : 0.7442

exact-matching test.other: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 2939/2939 [00:00<00:00, 21982.22 examples/s]
===== results for test.other =====
test.other BLEU   : 44.1736
test.other WER    : 0.2776
test.other CER    : 0.1154
test.other ROUGE1 : 0.7211
test.other ROUGE2 : 0.5286
test.other ROUGEL : 0.7205
"""
