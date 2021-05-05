# Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks (NIPS' 15)

## Abstract

object detection이란?

=> 사진 속의 물체가 어떤 위치에 존재하고 어떤 클래스로 분류되는지 알려줄 수 있는 작업



SPPnet과 Fast R-CNN이 나온 이후에 object detection에 대한 성능이 많이 증가함, 하지만 Fast R-CNN 같은 경우에도 Region Proposal에 많은 시간이 소요된다는 단점이 존재



그래서 이 논문에서는 Region Proposal Network를 따로 제안함(cpu -> gpu), 획기적으로 시간을 단축함

기존의 Fast R-CNN보다 빠르기 때문에 Faster R-CNN이라고 이름을 붙임

RPN network는 object가 어떠한 위치에 있는가 아닌가를 판단하게 해줌

RPN network는 end-to-end 방식으로 학습(gpu로 학습하기 때문에 가능), RPN network를 object의 위치와 크기를 파악하고 Fast R-CNN을 이용하여 어떤 클래스로 분류되는지 확인



## 1 INTRODUCTION

R-CNN은 기존에 계산 비용이 비싸다는 단점이 존재 -> sharing convolutions across proposal에 의해 개선

Selective search -> CPU환경에서 계싼이 이루어지므로 이미지당 2초의 시간이 소요됨, 1보다 작은 fps라 굉장히 느림

Edge box -> 이미지당 0.2초로 줄어들었지만 그래도 시간이 많이 소요됨

RPN network 제안 => 이미지당 10ms 정도의 시간 소요-> cost free

두 가지 기능 제공 -> regress region bounds & objectness score 계산

기존에는 pyramids of images나 pyramids of filters를 사용 -> 본 논문에서는 anchor boxes 사용함으로써 속도상의 이점을 얻음

RPN과 Fast R-CNN의 object detecion network 통합을 위해 학습을 번갈아가며 진행





## 3 Faster R-CNN

Region Proposal Network -> classifier

두 가지 모듈이 사용됨, fully convolutional network + Fast R-CNN detector

이를 합쳐서 Faster R-CNN으로 구성



<br/>



### 3.1 Region Proposal Networks

한 장의 입력을 이미지로 받아서 bounding box 형태로 output을 냄(각각의 bounding box는 물체가 그 위치에 존재하는지 안하는지 정보를 담을 수 있도록 함 *objectness score)

Fast R-CNN과 Feature map을 공유

ZF, VGG-16과 같은 아키텍쳐 사용 가능





![Faster1](Faster R-CNN Towards Real-Time Object Detection with Region Proposal Networks (NIPS' 15).assets/Faster1.PNG)

한 장의 이미지를 convolution layer에 넣어서 conv feature map을 도출

-> 다양한 크기를 가지는 anchor boxes를 이용하여 슬라이딩

-> 각 위치에 대해서 prediction 진행(기본적인 크기 3x3으로 256-d로 mapping)

-> classification layer와 regression layer로 물체가 존재하는지 아닌지를 판단

-> 존재한다면 어떤 위치에 존재하는지 width, height, x, y로 보다 정확하게 예측할 수 있도록 함



### 3.1.1 Anchors

각각의 sliding-window location에 대해서 region proposal을 예측함

-> 각 위치마다 k 개의 박스를 이용

-> 존재여부를 따지는 cls layer에서 2k score가 사용(softmax형태의 score)

3 scales와 3 aspect의 비율을 사용하기 때문에 총 9개의 anchors boxes를 사용함



### Translation-Invariant Anchors

이미지에 이동이 가해지더라도 translation-invariant property가 보장이 됨

위 그림에서 봤듯이 sliding window를 왼쪽 위 부터 오른쪽 아래까지 적용하였을 때 이미지에 translation이라는 변형이 가해진다고 볼 수 있다.

이러한 이동에 invariant하다는 것



### 3.1.2 Loss Function

각 anchor에 대해서 binary classification을 진행함

-> 여기서 positive label을 주는 경우는 두 가지가 있음

	1. highest Intersection-over-Union(IoU) 0.7이상이면 positive, 0.3이하이면 	negative



![loss](Faster R-CNN Towards Real-Time Object Detection with Region Proposal Networks (NIPS' 15).assets/loss.PNG)

![loss2](Faster R-CNN Towards Real-Time Object Detection with Region Proposal Networks (NIPS' 15).assets/loss2.PNG)



### 3.1.3 Training RPNs

RPN은 end-to-end 방식으로 학습이 가능

다만 모든 anchor에 대해서 loss를 계산하지 않음-> random하게 256개의 anchor만 sampling & positive anchor와 negative anchor의 비가 1:1이 되도록 함



### 3.2 Sharing Features for RPN and Fast R-CNN

기본적으로 Fast R-CNN의 detection network를 사용함 여기에 RPN network를 추가한 것이 본 논문에서 사용하는 network

이런 convolutional layer를 sharing하는 방법을 소개

1. *Alternating training: RPN을 학습하고 Fast R-CNN을 학습함 반복하며 번갈아가며 학습을 진행(본 논문에서 사용)
2. Approximate joint training: 네트워크 두 개를 하나의 네트워크처럼 묶어서 사용, 구현하기는 쉽지만 boxes' coordinates의 미분 값을 무시하기 때문에 어느정도 approximate된 결과를 얻음
3. Non-approximate joint training: box coordnates의 gradients를 포함시켜 backpropagation을 진행하는 방식, 이 논문에서 다루진 않는 방식



### 3.3 Implementation Details

각 anchor에 대해서 128x128, 256x256, 512x512 세 가지 scale, 1:1, 1:2, 2:1 세 가지 ratio이 적용된 서로 다른 anchor box를 사용 => anchor box의 크기가 고정되어있지만 regression을 통해 수정해나감

image boundaries를 cross하는 anchor boxes는 조심스럽게 접근할 필요가 있음

본 논문에서는 loss에 기여할 수 없게 무시하는 방식을 사용함