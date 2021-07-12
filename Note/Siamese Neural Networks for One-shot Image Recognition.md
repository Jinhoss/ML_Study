# Siamese Neural Networks for One-shot Image Recognition

[논문](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)

__One shot learning setting__ => 하나의 클래스 당 하나의 example만 주어져 있는 상태

target task와 관련된 domain-specific feature을 이용하면 문제를 쉽게 다룰 수 있지만, 새로운 domain이 주어졌을 때 예측능력이 떨어진다. 이 논문에서는 input들 간의 similarity를 계산하는 **siamese neural network**를 제안



## Approach

이 논문에서는 supervised metric-based 방식과 siamese neural network를 이용해 이미지에 대한 학습을 진행하고 학습된 feature들을 retrainig 없이  One shot learning에 이용한다.



