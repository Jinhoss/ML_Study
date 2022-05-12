# DiffCSE: Difference-based Contrastive Learning for Sentence Embeddings(NAACL 2022)



## Background

__Learning 'universal' sentence representation__

- 다양한 NLP task에 범용적으로 활용될 수 있는 임베딩 탐색
- 최근 여러 augmentation을 통해 positive pair를 구축하고, 어떠한 label 없이도 학습할 수 있는 contrastive learning 기반 방법론이 많이 연구됨
- SimCSE: Dropout을 통한 positive pair 구축
  - SimCSE는 hidden representation에 변형을 가하는 방법
  - 입력 테스트 자체는 변함 x
  - deletion, replacement 등의 방법은 입력 자체를 변형(텍스트의 의미가 바뀔 가능성 o)
  - 좋은 임베딩이라면 입력 테스트의 의미가 바뀔 때 마찬가지로 바뀌어야 함



## Related Work

__Equivariant contrastive learning__

- 본 논문(DiffCSE)은 equivariant contrastive learning의 한갈래라고 볼 수 있음
- 많은 self-supervised learning 방법론에서는 data에 어떠한 변형을 가하고 invariant한 특성을 갖도록 학습을 유도함(SimCLR 등)
- Equivariance는 invariance의 일반화된 개념이며, 직관적으로는 input에 적용된 변형의 정도만큼 output feature representation도 변형된다는 개념

$$
\forall x: f(T_g(x))=T'_g(f(x))
$$

 

## Method

__Difference-based learning__

- 앞선 방법론의 주장을 받아들여, 두 가지 loss를 혼합하여 사용

1. Insensitive transformation:dropout-based augmentation(SimCSE와 같음)

2. Sensitive transformationion:MLM-based augmentation

   ![diffcse1](DiffCSE Difference-based Contrastive Learning for Sentence Embeddings(NAACL 2022).assets/diffcse1.PNG)

- 왼쪽이 Standard SimCSE, 오른쪽이 Conditional difference prediction model
- Sentence Encoder에는 BERT나 RoBERT를 사용, Generator에는 DistilBERT나 DistilRoBERTa
- 오른쪽 모델은 학습할 때만 사용, 실제 테스트에서는 왼쪽에 있는 Sentence Encoder만 사용



## Ablation studies

- Contrastive loss 매우 중요(84.56=>54.48)
- Same sentence로 MLM이 아닌 next sentence로 augmentation => (82.91)
- Other conditional pretraining task => (83.41)

- MLM, insert, delete 중에 mlm이 전반적으로 성능이 좋음
- 바로 CLS 토큰을 사용하는 것보다 two-layer pooler와 batchnorm을 사용하는 것이 좋음
- Generator로 DistilBERT를 사용하는 것이 효율적이고 좋은 성능
- 30% 정도의 masking이 좋은 성능