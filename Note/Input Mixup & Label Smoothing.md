# Input Mixup & Label Smoothing

## 1. Mixup Training

학습을 진행할 때 랜덤하게 두 개의 샘플 (x_i, y_i), (x_j, y_j)를 뽑아서 일정한 비율로 섞어 새로운 (x, y)를 만들어 학습에 사용한다.

데이터를 섞음으로써 데이터 증강효과, 과적합방지

<br/>



## 2. Label Smoothing

이미지는 건드리지 않고 레이블만 변경 => 일반화 성능을 높임

정답 레이블에 대해 100%의 확률을 부여하지 않음, 일정 비율을 깎아 다른 레이블에 비율을 부여(soft)



두 방법 모두 over confident하지 않도록하여 정규화 효과를 누릴 수 있음

Label 같은 경우에는 사람이 Label하는 경우 Labeling이 잘못되어 있는 경우가 있음, 그렇기 때문에 soft labeling을 통해 이를 어느정도 완화시킬 수 있음

<br/>





#### CIFAR10 데이터 기준 Mixup과 Label Smoothing 사용시 정확도 약 94% -> 95%



