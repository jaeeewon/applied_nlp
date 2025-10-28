import os
from g2p_en import G2p
from tqdm import tqdm

librispeech_subsets = [
    "dev-clean",
    "dev-other",
    "test-clean",
    "test-other",
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
]


def get_librispeech_transcriptions(root_dir: str, subsets: list[str], g2p: G2p):
    ls = []  # [id, text, phonemes][]
    for subset in tqdm(subsets):
        for i in tqdm(os.listdir(f"{root_dir}/{subset}")):
            if not i.isdigit():
                continue
            for ii in os.listdir(f"{root_dir}/{subset}/{i}"):
                if not ii.isdigit():
                    continue
                path = f"{root_dir}/{subset}/{i}/{ii}"
                with open(f"{path}/{i}-{ii}.trans.txt", encoding="utf-8") as f:
                    for line in f:
                        id, text = line.strip().split(" ", maxsplit=1)
                        phonemes = g2p(text)
                        ls.append([id, text, phonemes])
    return ls


if __name__ == "__main__":
    g2p = G2p()
    transcriptions = get_librispeech_transcriptions(
        "/home/jpong/Workspace/jaeeewon/LibriSpeech",
        subsets=["dev-clean"],
        g2p=g2p,
    )

    print(len(transcriptions))
    print(transcriptions[0])
