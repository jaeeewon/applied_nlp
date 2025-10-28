# 프로젝트 제안서

- Language & AI융합전공 202402050 최재원

---

## 데이터셋

- **dataset:** [*/*](https://huggingface.co/datasets/*/*)
- **subset:** *
- **split:** *

### 분석

- 데이터 크기, 컬럼명, 예시 데이터 3개, 라이선스, 언어, 목적 등

---

## 모델 선택

- **model:** [tencent/Hunyuan-MT-7B](https://huggingface.co/tencent/Hunyuan-MT-7B)
- **technical report:** [arxiv](https://arxiv.org/pdf/2509.05209)
- 모델 정보
    - 모델 이름: Hunyuan-MT-7B
    - 특징: 
    - 학습 데이터: 
    - 토크나이저: 
- 모델 선택 이유
    - multilingual MT 모델의 En2Ko BLEU4 score를 높이는 실험을 진행하고자 함
    - In the WMT25 competition, the model achieved first place in 30 out of the 31 language categories it participated in.
    - It primarily supports mutual translation among 33 languages, including five ethnic minority languages in China.