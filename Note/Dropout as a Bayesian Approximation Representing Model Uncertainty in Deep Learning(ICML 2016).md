# Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning(ICML 2016)



## Background



__Monte Carlo integration__
$$
log\int{p(y^*|x^*, \omega)q(\omega)d\omega}\\\\
log({1\over T}\sum^T_{t=1}p(y^*|x^*, \omega_t))
$$

- 복잡한 형태의 함수의 기대 값을 직접 계산하기 어려울 때 사용
- Random sampling을 통해 연속 공간의 적분 값을 근사



__Dropout in Neural Network__

- 학습 과정에서 dropout ratio 만큼의 node를 랜덤하게 제거
- overfitting 방지를 위해 사용



__Bayesian modeling__
$$
p(y^*|x^*, X, Y) = \int p(y^*|x^*, \omega)p(\omega|X,Y)d\omega
$$
X, Y 는 학습 데이터, x *, y *는 테스트 데이터

- Bayesian modeling의 inference는 training time의 optimization을 포함
- 설명력을 가질 수 이쓰나, 많은 계산량을 필요로 함



__Bayesian neural network__

![bayesian](Dropout as a Bayesian Approximation Representing Model Uncertainty in Deep Learning(ICML 2016).assets/bayesian.PNG)

- Posterior를 근사할 함수를 Bayesian neural network로 사용
- Neural network의 weight를 distribution으로 나타냄



## Introduction

__Deep Learning & Bayesian probability__

- Deep learning model의 softmax output으로는 uncertainty를 나타낼 수 없음
- Bayesian probability theory를 사용하면 수학적으로 uncertainty를 추론할 수 있지만 많은 계산량을 필요로 함
- Neural network에 dropout을 적용하는 것이 Gaussian process(GP)의 Bayesian approximation으로 해석될 수 있음



## Dropout as Bayesian Approximation

__Covariance function__
$$
K(x,y) = \int p(w)p(b)\sigma(w^Tx + b)\sigma(w^Ty+b)dwdb
$$


__MC estimator__
$$
\hat{K}(x, y) = {1 \over K}\sum^K_{k=1}\sigma(w^T_kx+b_k)\sigma(w^T_ky+b_k)
$$


__Deep Gaussian Process model__
$$
\phi(x,W_1,b) = \sqrt{1\over K}\sigma(W^T_1x+b)\\\\
p(y|x,X,Y)=\int p(y|x,\omega)p(\omega|X,Y)d\omega\\\\
p(y|x,\omega) = \Nu(y;,\hat{y}(x,\omega),\tau^{-1}I_D)\\\\
\hat{y}(x,\omega=\{{W_1, ...,W_L}\})=\sqrt{1 \over K_L}W_L \sigma(...\sqrt{1 \over K_1}W_2\sigma(W_1x+m_1)...)
$$
