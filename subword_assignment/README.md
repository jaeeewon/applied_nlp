# 1. 서브워드 비교해보기

## 목표
- 같은 데이터에 대해 서브워드 학습 데이터, 방법, 하이퍼파라미터를 변경했을 때, 차이점을 이해하기

## 요구사항
- 원하는 데이터 본인이 선택하기
- BPETrainer 써보기

## 참고
- **BPE model:** [huggingface - tokenizers.models.BPE](https://huggingface.co/docs/tokenizers/en/api/models#tokenizers.models.BPE)
- **BPE trainer:** [huggingface - tokenizers.trainers.BpeTrainer](https://huggingface.co/docs/tokenizers/api/trainers#tokenizers.trainers.BpeTrainer)

## 최대 단어 수 변경하기
- `vocab_size = 16,000`
- `vocab_size = 1,600`

## 제출물
- 서브워드 학습 데이터, 방법, 하이퍼파라미터 변경(최대 단어 수 변경)에 따른 차이점 보고서
- 구현 `ipynb` 파일 제출