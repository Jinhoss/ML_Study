# TabNet: Attentive Interpretable Tabular Learning(ICLR 2020)

[논문](https://openreview.net/forum?id=BylRkAEKDH)

## 1 INTRODUCTION

DNN은 Image, Text, Audio와 같은 비정형 데이터를 이해하는데 강력하다.( ex: image의 경우에는 resnet이 좋은 성능을 내고 있다) 

하지만 하지만 가장 일반적인 데이터의 유형인 정형데이터(tabular data or structured data)를 이해하는 표준 NN 아키텍처 연구는 그렇지 못하다. 대신에 변형된 ensemble decision tree가 많이 사용된다.

(ex: lgbm, catboost)

계산 비용이 증가하더라도 고성능 딥러닝 NN 아키텍처가 필요하다.

기존의 문제점을 개선한(역전파 x, streaming data 분석 취약) 모델을 제안(FE, feature selection, 설명가능한 모델)



### Tabnet의 핵심 & 기여도

1. flextible representation을 배우고 flexible integration into end-to-end 딥러닝을 하기 위해선 기존 정형 데이터 분석 방법과는 달리 TabNet은 FE없이 원데이터를 사용하여 학습

2. 성능과 설명 가능성을 높이기 위해서, seqautial attention mechanism을 활용

3. 정형 데이터 학습에 대한 두 가지 속성을 이끌어 냈다.

   (1) TabNet은 기존의 정형 데이터 분석 모델보다 분류 문제와 회귀 문제에서 뛰어난 성능을 보인	    다.

   (2) TabNet은 2가지 종류의 해석이 가능

    	- input 변수들과 그들을 결합하는 방법의 중요성을 시각화 하기 위한 local interpretability
    	- 학습된 모델에서 각 input feature의 기여도를 표현하는 global interpretability

   

![tabnet1](TabNet Attentive Interpretable Tabular Learning(ICLR 2020).assets/tabnet1.png)



위 그림과 같이 -> 방향으로 sequential feature selection 진행 & feedback 진행하며 학습



## 3 TABNET MODEL

#### 3.1 PRINCIPLES



![tabnet principles](TabNet Attentive Interpretable Tabular Learning(ICLR 2020).assets/tabnet principles.png)



전통적인 NN block과 2-d manifold를 사용하여 decision tree-like 분류의 설명한 그림

input들에게 multiplicative sparse mask를 배치함으로써 관련 feature들이 선택된다.(x1, x2는 인풋차원, a 와 d는 상수)

선택된 feature들은 선형적으로 변형되고 그 후에 bias를 추가(boundaries 표현)

Relu 함수는 영역들을 제로화함으로써 영역 선택을 할 수 있다. 다중 영역 통합은 합 연산을 기반으로 한다.

C1, C2가 크면 클수록 decision boundary는 softmax로 더 sharper해질 수 있다.



#### 3.2 OVERALL ARCHITECTURE

![tabnet architecture](TabNet Attentive Interpretable Tabular Learning(ICLR 2020).assets/tabnet architecture.png)

위 그림은 TabNet의 아키텍처, batchnormalization을 따르는 fully-connected layer을 사용

Tabular data input들은 수치형 변수와 범주형 변수로 구성

저자는 오로지 batch normalization을 적용

저자는 각 decision step에서  D차원 변수들 f ∈ B×D을 통과한다. (for B = batch-size) 

전반적으로 모델의 성능 대부분이 하이퍼파라미터에 민감하게 반응하지 않음



#### Feature selection

soft selection을 위해 학습 가능한 sparse mask를 배치(M [i] ∈ B×D)

a[i-1] 처리딘 단계로부터 전처리된 변수들을 사용한 마스크를 얻기 위한 대안적인 transformer를 사용



#### TabNet 아키텍처 각 결정 단계

1. feature transformer
2. attentive transformer
3. feature masking으로 구성



#### Feature Processing

feature transformer을 사용하여 선택된 변수들을 처리

![equation3](TabNet Attentive Interpretable Tabular Learning(ICLR 2020).assets/equation3.png)



![equation4](TabNet Attentive Interpretable Tabular Learning(ICLR 2020).assets/equation4.png)



빠른 학습을 위해 larger batch size 사용





![equation5](TabNet Attentive Interpretable Tabular Learning(ICLR 2020).assets/equation5.png)

