# Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift(2015)



# Intro

- SGD의 문제점: 레이어를 여러층으로 쌓고 있기 때문에 발생하는 문제점, 각 layer에서는 input값의 분포가 지속적으로 변화하므로 layer는 계속 변화하는 분포에 적응해야하는 문제점이 있다.(covariate shift)

  => 각 layer에 들어가는 input들의 분포를 고정한다면 각 layer의 파라미터가 학습시마다 변경되는 input들의 분포에 적응하여 변화할 필요가 없음

- Sigmoid와 같은 활성함수들의 미분값이 0 근처에서 멀어질수록 작아지는 문제때문에 발생하는 saturation해결 가능

  - 이전에는 주의깊은 parameter 초기화, 작은 lr을 사용하였으나 batch normalization을 통하여 input의 분포를 제어하고 이를 통해 saturation을 방지가능