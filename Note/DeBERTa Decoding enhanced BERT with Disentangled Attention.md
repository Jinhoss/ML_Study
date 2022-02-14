# DeBERTa: Decoding enhanced BERT with Disentangled Attention

## Background

__Roberta__

- Model 학습시간 증가, Batch 사이즈(약 32배)를 늘리고 train data(약 10배) 증가
  - 데이터 양을 늘릴수록 Downstream task의 성능이 증가함
- Next sentence prediction 제거
- Longer sequence를 추가(512를 넘지 않는 선에서)
- Masking pattern을 dynamic하게 해줌
  - BERT는 pretrain 전에 미리 masking을 진행, 학습이 진행될 때 똑같은 token이 masking된다는 문제가 있음(bias)
  - 똑같은 데이터에 대해 masking을 10번 다르게 적용하여 학습(dynamic masking)
  - input이 들어갈 때마다 masking 진행
- 구조적 변화가 전혀 없으나 타 모델 대비 성능이 좋음



## Prerequisite

__Disentangled__

- 이미지 생성 모델(cf. infoGAN)
- 서로 뒤얽혀 있는 특징 요소들을 독립적으로 풀어서 표현
- 데이터의 다양성을 설명하는 latent요소들을 분리하여 표현함으로써 interpretability를 높임
- ex) 얼굴 => 안경, 수염, 머리, 눈, 나이 

- 좀 더 일반적으로 표현하고자 하는 어떤 하나의 개념을 여러가지 sub 개념으로 분해하여 표현하는 것으로 이해가 가능하다.



__Positional Embedding(Encoding)이란?__

- 단어의 위치 정보를 특정 차원의 벡터로 표현하는 것
- Transformer에서는 각 단어의 임베딩 벡터에 위치 정보들을 더하여 모델의 입력으로 사용
- Positional Embedding 벡터는 학습하여 사용할 수도 있고, 특정 길이에 따라 고정된 값을 사용할 수도 있음(Non-Trainable)



| Absolute Positional Embedding                                | Relative Positional Embedding                                |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| 어떤 입력 문장이더라도 각 단어의 위치의 Positional Embedding 값은 동일한 값이 사용됨 | 특정 위치의 단어를 기준으로 일정 길이(윈도우) 이내에 위치한 단어와의 상대적 거리 관계 정보를 반영 |
| 보통 모델의 Input Embedding에 Positional Embedding을 더해서 사용 | Attention을 구하는 Matrix 연산 과정에서 Positional Embedding을 활용 |
| 각 단어의 position 값이 index가 되어 그에 해당하는 position embedding값을 position embedding matrix에서 뽑아서 사용 | 각 단어 간의 상대적인 위치 차이가 index로 사용 됨            |
| 각 토큰의 절대적 위치 정보, 절대적으로 떨어진 거리 정보를 파악할 수 있음 => distance | 짝지어진 토큰 간의 상대적 위치 정보를 파악할 수 있음 => direction and distance |

