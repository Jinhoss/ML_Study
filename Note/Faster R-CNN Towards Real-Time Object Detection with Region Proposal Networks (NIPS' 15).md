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