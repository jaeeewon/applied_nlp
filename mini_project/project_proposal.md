# 프로젝트 제안서

- Language & AI융합전공 202402050 최재원

---

## 데이터셋

### 구축

- LibriSpeech train-clean/other set 준비
- transcription을 g2p_en을 이용해 phonemes 획득
- {id, text, phonemes}의 형태로 huggingface에 데이터셋 업로드
- 학습 시점에 noise를 입히고 데이터셋을 업데이트

- **dataset:** [jaeeewon/librispeech_phonemes](https://huggingface.co/datasets/*/*)
- **subset:** *
- **split:** train

### 분석

- 데이터 크기
    - *
- 컬럼명
    - *
- 예시 데이터 3개
    - *
- 라이선스
    - *
- 언어
    - *
- 목적
    - *

---

## 모델 선택

- **model:** [facebook/bart-large](https://huggingface.co/facebook/bart-large)
- **paper:** [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension (arxiv:1910.13461)](https://arxiv.org/abs/1910.13461)
- 모델 정보
    - 모델 이름
        - BART (Bidirectional and Auto-Regressive Transformers)
    - 특징
        - Pretraining has two stages (1) text is **corrupted with an arbitrary noising function**, and (2) a sequence-to-sequence model is learned to **reconstruct the original text**.
        - A key advantage of this setup is the noising flexibility; arbitrary transformations can be applied to the original text, **including changing its length**.
        - BART also provides a **1.1 BLEU increase over a back-translation** system for machine translation, with only target language pretraining.
        - BART uses a **standard Tranformer-based neural machine translation architecture** which, despite its simplicity, can be seen as generalizing BERT (due to the bidirectional encoder), GPT (with the left-to-right decoder), and many other more recent pretraining schemes (see Figure 1).
        - Because BART has an autoregressive decoder, **it can be directly fine tuned for sequence generation tasks** such as abstractive question answering and summarization.
    - 학습 데이터
        - We use the same pre-training data as [Liu et al. (2019)](https://arxiv.org/abs/1907.11692), consisting of 160Gb of news, books, stories, and web text.
            - We consider five English-language corpora of varying sizes and domains, totaling over 160GB of uncompressed text. We use the following text
            corpora:
                - BOOKCORPUS (Zhu et al., 2015) plus English WIKIPEDIA. This is the original data used to train BERT. (16GB).
                - CC-NEWS, which we collected from the English portion of the CommonCrawl News dataset (Nagel, 2016). The data contains 63 million English news articles crawled between September 2016 and February 2019. (76GB after filtering).
                    - We use news-please (Hamborg et al., 2017) to collect and extract CC-NEWS. CC-NEWS is similar to the REALNEWS dataset described in Zellers et al. (2019).
                - OPENWEBTEXT (Gokaslan and Cohen, 2019), an open-source recreation of the WebText corpus described in Radford et al. (2019). The text is web content extracted from URLs shared on Reddit with at least three upvotes. (38GB)
                    - The authors and their affiliated institutions are not in any way affiliated with the creation of the OpenWebText dataset.
                - STORIES, a dataset introduced in Trinh and Le (2018) containing a subset of CommonCrawl data filtered to match the story-like style of Winograd schemas. (31GB).
    - 토크나이저
        - BPE
            - Documents are tokenized with the same byte-pair encoding as GPT-2 (Radford et al., 2019). 
- 모델 선택 이유
    - 우리의 목표와 유사하게 학습됨
        - **noised phonemes**(input)를 homonym도 구분하며 **sentence**(output)로 복구
            - 방대한 데이터로 학습하였기에, homonym을 구분할 수 있을 것이라고 생각함
            - **Catastrophic Forgetting**을 막고자 LoRA를 사용해볼 생각
                - 잘 되지 않는다면, noised phonemes와 sentence의 기존 데이터셋에 추가로 BART가 학습한 방식으로도 학습해볼 생각
    - 우리의 목표를 수행할 수 있을 정도의 견고한 모델 구조
    - 우리의 목표에 적절한 모델 구조
    - 우리의 목표를 쉽게 구현할 수 있는 모델 구조
