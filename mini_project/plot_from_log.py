import numpy as np, matplotlib.pyplot as plt, os, json


def plot_curves(train_losses, val_losses, filename, train_any=None, val_any=None, any_name=""):
    epochs = np.arange(1, len(train_losses) + 1)

    plt.figure(figsize=(10, 4))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Steps (2k)")
    plt.ylabel("Loss")
    plt.title("Loss over Steps (2k)")
    plt.legend()
    plt.grid(True)

    if any_name:
        # any
        plt.subplot(1, 2, 2)
        if train_any:
            plt.plot(epochs, train_any, label=f"Train {any_name}")
        if val_any:
            plt.plot(epochs, val_any, label=f"Val {any_name}")
        plt.xlabel("Steps (2k)")
        plt.ylabel(any_name)
        plt.title(f"{any_name} over Steps (2k)")
        plt.legend()
        plt.grid(True)

    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    plt.savefig(os.path.join("plots", filename), dpi=200)
    plt.close()


def read_log_file(log_file):
    train_loss, eval_loss, eval_bleu, eval_wer, eval_cer, eval_rougeL = [], [], [], [], [], []

    with open(log_file, "r") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if line.startswith("{'eval_loss"):
                log = json.loads(line.replace("'", '"'))
                eval_loss.append(log["eval_loss"])
                eval_bleu.append(log["eval_bleu"])
                eval_wer.append(log["eval_wer"])
                eval_cer.append(log["eval_cer"])
                eval_rougeL.append(log["eval_rougeL"])
                log = json.loads(lines[i + 1].replace("'", '"'))
                train_loss.append(log["loss"])

    return train_loss, eval_loss, eval_bleu, eval_wer, eval_cer, eval_rougeL


if __name__ == "__main__":
    log_file = "251205_1214.log"
    train_loss, eval_loss, eval_bleu, eval_wer, eval_cer, eval_rougeL = read_log_file(log_file)

    plot_curves(train_loss, eval_loss, log_file.replace(".log", "_loss_wer.png"), val_any=eval_wer, any_name="WER")
    plot_curves(train_loss, eval_loss, log_file.replace(".log", "_loss_cer.png"), val_any=eval_cer, any_name="CER")
    plot_curves(train_loss, eval_loss, log_file.replace(".log", "_loss_bleu.png"), val_any=eval_bleu, any_name="BLEU")
    plot_curves(train_loss, eval_loss, log_file.replace(".log", "_loss_rougeL.png"), val_any=eval_rougeL, any_name="ROUGE-L")
