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

BERT의 output에 pooling layer를 추가, 논문에서는 pooling 전략 3가지에 대해서 소개

1. CLS 토큰의 결과를 사용
2. 모든 output vector를 평균하여 사용
3. output vector의 max-over-time을 계산해서 사용

기본적으로 2번 방법을 사용

BERT를 fine-tuning하기 위해 siamse network/ triplet/network를 추가, 이를 이용해서 weight를 조정

모델 구조는 학습 데이터에 따라 다름



#### 1) Classification Objective Function

![sentencebert1](../../sentencebert1.PNG)

각 문장에서 생성되는 벡터 u, v 그리고 그 거리를 concat하여 입력, loss는 cross-entropy로 설정



#### 2) Regression Objective Function

![sbert2](../../sbert2.PNG)

코사인 유사도를 계싼, MSE를 loss로 사용



#### 3) Triplet Objective Function

![sbert3](../../sbert3.PNG)

anchor/positive/negative 문장 a, p, n에 대해 triplet loss는 위와 같다. s는 각문장 의 임베딩이고

epsilon은 margin이다, 이 논문에서 Euclidean 거리를 단위로 epsilon = 1로 설정



## Ablation Study

... 중략, Classification objective function을 학습할 때에는 pooling 전략의 차이는 큰 영향이 없음

어떤 것을 concat하느냐에 따라 성능의 차이가 큼, inferSent나 Universal Sentence Encoder는 u*v를 추가했는데, SBERT에서는 이를 추가하면 오히려 성능이 감소

가장 중요한 요소는 두 임베딩의 거리이다. 비슷한 벡터는 가깝게, 비슷하지 않은 벡터는 멀게 하도록 도와줌



반면, Regression objective function을 학습할 때에는 pooling 전략이 큰 영향을 미침

