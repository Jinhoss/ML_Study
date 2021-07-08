# Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks

[논문](https://arxiv.org/pdf/1908.10084.pdf)

## Introduction

Sentence-BERT => BERT에서 siamse network와 triplet network를 활용

semantically meanigful sentence embedding을 구할 수 있음

기존의 BERT로 적용하기 힘든 과제들 => large-scale semantic similarity comparison, clustering, information retrieval via semantic search.. 



clustering이나 semantic search에서 보통 접근법이 vector embedding으로 공간상 유사도를 계산

-> BERT에서는 forward 한 후 평균값을 이용하거나 CLS 토큰의 값을 이용함 하지만 GloVe 임베딩에 평균을 취하는 것보다 결과가 좋지 않음



하지만 SBERT에서...

1. siamese network는 input sentence에 대한 일정한 크기의 vector 출력이 가능

2. 거리를 이용한 공간상 유사도 측정방법으로 유사한 의미를 지닌 문장을 찾을 수 있음

   즉, semantic similarity나 clustering같은 목적으로 활용이 가능

3. BERT는 10000개의 문장 집합에서 유사 문장을 찾는데 65시간이 걸리지만 SBERT는 고정된 크기의 벡트 10000개에서 유사한 문장을 찾는데 5초가 걸림, cosine similarity는 0.01초

이러한 차이가 발생하는 이유는 BERT의 경우에는 전체 문장의 개수가 n이라고 했을 때 각 문장마다

n-1만큼의 BERT network를 거쳐야하기 때문,

반면 SBERT는 모두 고정 크기의 벡터로 변환해서 저장하고, 이 벡터들로 similartiy를 구해서 argmax한다. 

각 문장이 network를 한 번 거치므로 효율성의 차이가 상당히 크다



## Related Work

- BERT
- Sentence Embeddings

## Model





