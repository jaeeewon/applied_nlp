# torchrun --nproc_per_node=4 train.py

import evaluate
import numpy as np
import json
import argparse
from datasets import load_dataset, concatenate_datasets
from transformers import (
    BartTokenizerFast,
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EarlyStoppingCallback,
    TrainerCallback,
)
from typing import Any
from noti import send_noti

MODEL_NAME = "facebook/bart-large"
DATASET_NAME = "jaeeewon/librispeech_phonemes"

USE_WORD_BOUNDARY = False
WORD_BOUNDARY_TOKEN = "<WB>"
PHONEME_LAMBDA = lambda x: x  # f"<PH_{x}>"

MAX_LENGTH_PHONEME = 512
MAX_LENGTH_GRAPHEME = 128


bleu_metric = evaluate.load("sacrebleu")
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")
rouge_metric = evaluate.load("rouge")


def build_tokenizer_and_model():
    tokenizer = BartTokenizerFast.from_pretrained(MODEL_NAME)
    model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)

    raw_datasets = load_dataset(DATASET_NAME)

    train_raw = concatenate_datasets(
        [
            raw_datasets["train.clean.100"],
            raw_datasets["train.clean.360"],
            raw_datasets["train.other.500"],
        ]
    )

    phoneme_set = set()
    for ex in train_raw:
        for ph in ex["phonemes"]:
            if ph != " ":
                phoneme_set.add(PHONEME_LAMBDA(ph))

    phoneme_list = sorted(list(phoneme_set))
    print(f"{len(phoneme_list)} unique phonemes")

    phoneme_list.append(WORD_BOUNDARY_TOKEN)

    added = tokenizer.add_tokens(phoneme_list)
    model.resize_token_embeddings(len(tokenizer))

    print(f"added and resized {added} phoneme tokens to tokenizer")

    return tokenizer, model, raw_datasets


def preprocess_function_builder(tokenizer):
    def preprocess_function(examples: dict[str, list[Any]]) -> dict[str, Any]:
        phoneme_sequences = []
        texts = []

        for phonemes, text in zip(examples["phonemes"], examples["text"]):
            tokens = [(WORD_BOUNDARY_TOKEN if ph == " " else PHONEME_LAMBDA(ph)) for ph in phonemes]

            if not USE_WORD_BOUNDARY:
                tokens = [ph for ph in tokens if ph != WORD_BOUNDARY_TOKEN]

            ph_str = " ".join(tokens)
            phoneme_sequences.append(ph_str)

            texts.append(text.lower())

        model_inputs = tokenizer(phoneme_sequences, padding=False, truncation=True, max_length=MAX_LENGTH_PHONEME)
        # print(len(model_inputs.input_ids[0]), phoneme_sequences[0], model_inputs.input_ids[0])

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(texts, padding=False, truncation=True, max_length=MAX_LENGTH_GRAPHEME)

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    return preprocess_function


class NotiCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return
        if logs is None:
            return

        step = state.global_step
        eval_loss = logs.get("eval_loss", None)

        if eval_loss is None:
            return

        send_noti(
            title=f"BART FFT WB={USE_WORD_BOUNDARY} step={step}",
            message=json.dumps(logs),
        )


def build_compute_metrics(tokenizer):
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        preds = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        for i in range(3):
            print("pred :", decoded_preds[i])
            print("label:", decoded_labels[i])

        bleu = bleu_metric.compute(
            predictions=decoded_preds,
            references=[[l] for l in decoded_labels],
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

        return {
            "bleu": bleu["score"],
            "wer": wer,
            "cer": cer,
            "rouge1": rouge["rouge1"],
            "rouge2": rouge["rouge2"],
            "rougeL": rouge["rougeL"],
        }

    return compute_metrics


def train(output_dir: str):
    tokenizer, model, raw_datasets = build_tokenizer_and_model()

    train_raw = concatenate_datasets(
        [
            raw_datasets["train.clean.100"],
            raw_datasets["train.clean.360"],
            raw_datasets["train.other.500"],
        ]
    )
    eval_raw = concatenate_datasets(
        [
            raw_datasets["dev.clean"],
            raw_datasets["dev.other"],
        ]
    )

    preprocess_function = preprocess_function_builder(tokenizer)

    columns_to_remove = ["id", "text", "phonemes"]

    tokenized_train = train_raw.map(
        preprocess_function,
        batched=True,
        remove_columns=columns_to_remove,
        desc="tokenizing train set",
    )
    tokenized_eval = eval_raw.map(
        preprocess_function,
        batched=True,
        remove_columns=columns_to_remove,
        desc="tokenizing dev set",
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    compute_metrics = build_compute_metrics(tokenizer)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=False,
        evaluation_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        eval_steps=2000,
        save_steps=2000,
        save_total_limit=3,
        logging_steps=10,
        learning_rate=1e-4,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=48,
        num_train_epochs=30,
        warmup_ratio=0.1,
        weight_decay=0.01,
        predict_with_generate=True,
        bf16=True,
        fp16=False,
        gradient_accumulation_steps=1,
        label_smoothing_factor=0.1,
        report_to=["none"],
        generation_max_length=MAX_LENGTH_GRAPHEME,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[NotiCallback()],
    )

    trainer.train()
    eval_metrics = trainer.evaluate()
    print("evaluation on dev:", eval_metrics)

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"model and tokenizer saved to {output_dir}")


def eval_model(output_dir: str):
    tokenizer, model, raw_datasets = build_tokenizer_and_model()
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    splits = [
        "dev.clean",
        "dev.other",
        "test.clean",
        "test.other",
    ]

    preprocess_function = preprocess_function_builder(tokenizer)
    compute_metrics = build_compute_metrics(tokenizer)

    tokenized_splits = {}
    columns_to_remove = ["id", "text", "phonemes"]

    for split_name in splits:
        tokenized = raw_datasets[split_name].map(
            preprocess_function,
            batched=True,
            remove_columns=columns_to_remove,
            desc=f"tokenizing {split_name}",
        )
        tokenized_splits[split_name] = tokenized

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        do_train=False,
        do_eval=True,
        per_device_eval_batch_size=64,
        predict_with_generate=True,
        generation_max_length=MAX_LENGTH_GRAPHEME,
        bf16=True,
        fp16=False,
        report_to=["none"],
        dataloader_drop_last=False,
    )

    dummy_first_split = splits[0]
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=None,
        eval_dataset=tokenized_splits[dummy_first_split],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    for split_name in splits:
        split_key = split_name.replace(".", "_")

        metrics = trainer.evaluate(
            eval_dataset=tokenized_splits[split_name],
            metric_key_prefix=split_key,
        )

        if trainer.is_world_process_zero():
            print(f"===== results for {split_name} =====")
            print(f"{split_name} BLEU   : {metrics[f'{split_key}_bleu']:.4f}")
            print(f"{split_name} WER    : {metrics[f'{split_key}_wer']:.4f}")
            print(f"{split_name} CER    : {metrics[f'{split_key}_cer']:.4f}")
            print(f"{split_name} ROUGE1 : {metrics[f'{split_key}_rouge1']:.4f}")
            print(f"{split_name} ROUGE2 : {metrics[f'{split_key}_rouge2']:.4f}")
            print(f"{split_name} ROUGEL : {metrics[f'{split_key}_rougeL']:.4f}")
            print()

    if trainer.is_world_process_zero():
        print("sucessfully finished evaluation for all splits")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["train", "eval"], default="eval", help="mode: train or eval")
    parser.add_argument("--output_dir", type=str, required=True, help="output directory")
    args = parser.parse_args()

    mode = args.mode

    if mode == "train":
        train(args.output_dir)
    elif mode == "eval":
        eval_model(args.output_dir)
