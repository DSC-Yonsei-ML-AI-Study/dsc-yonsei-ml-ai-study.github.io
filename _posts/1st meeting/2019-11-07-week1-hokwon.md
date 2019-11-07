---
author : Hokwon Lee
laytout : posts
title : "[호권/1st meetup] Intro to ML 키워드 조사"
---

## What is ML?
  기계학습으로 인간이 하나부터 열까지 직접 코드를 지정해 주는 것이 아닌 학습할 무언가를  기계에 주고 이걸 가지고 스스로 학습하는 기계이다.

## ML vs Rule-based
  ML은 학습을 통해 규칙이 정해지지 않아도 공통점을 찾아 동작을 실행하는 것이고 Rule-based는 정해진 규칙에 따라 동작을 실행하는 것이다.
  이 둘의 차이는 규칙을 정해 동작을 하는가? 규칙이 정해지지 않아도 동작을 하는가?이다.
   
## AI, ML, Deep Learning
  - AI는 Artificial Intelligence의 약자로 시스템에 의해 만들어진 지능으로 기계가 경험을 통해 학습하고 새로운 입력 내용에 따라 기존 지식을 조정하며 사람과 같은 방식으로 과제를 수행할 수 있도록 지원하는 기술이다.
  - ML은 Machine Learning의 약자로 AI의 한 분야로 인간이 하나부터 열까지 직접 코드를 지정해 주는 것이 아닌 학습할 무언가를  기계에 주고 이걸 가지고 스스로 학습하는 기계이다.
  - Deep Learning은 ML의 한 갈래로 인공 신경망의 새로운 이름이다. ML의 경우 기계가 학습하기 위해 주어진 데이터(예시)에서 특징을 추출하는 과정에서 사람이 개입, Deep Learning은 데이터 자체에서 중요한 특징을 기계 스스로 처음부터 끝까지 학습하는 기술이다.

## Type of ML (Classification, Clustering, Regression, Sequence Prediction)
  1. Classification(분류)
    - Supervised learning 지도학습의 일종으로 기존에 존재하는 데이터의 카테고리 관계를      파악하고, 새롭게 관측된 데이터의 카테고리를 스스로 판별하는 과정.
    - 특징 : 선형성
     - 예) 스팸 필터 
  2. Clustering(군집화)
     - Unsupervisoed Learning(비교사학습)의 기법 중 하나로 데이터를 비슷한 것끼리 묶는 것.
     - Quantization(양자화) 또는 Coding(코딩)이라고 부르기도 한다.
  3. Regression(회귀 분석)
    - 연속된 값을 예측하는 문제.
    - 어떤 패턴, 트랜드, 경향을 예측할 때 사용
    - 예) 공부시간에 따른 전공 시험 점수 예측
  4. Sequence Prediction
    - 히스토리 sequence 정보를 사용하여 sequence 다음 값을 예측하는 것.


##  Kind of Bias (Interaction bias, latent bias, selection bias)
  1. Interaction bias
    - 한쪽으로만 치우친 편향으로 다른 쪽으로는 값이 나오지 않는 ML
    예) 사람을 남자라고만 학습시키면 여자가 나왔을 경우 사람이라 값이 나오지않음

  2. latent bias
    - 잘못된 아이디어와 상관관계를 갖는 편향, 
    - 예) 의사를 학습시킬 때 남자의 이미지만 가지고 하면, 의사를 보여줄 때 남성의사만 
    보여준다.
  3. selection bias
    - 특정 집단을 집중적으로 선택하는 것에서 생기는 편향 
    -  다른 집단에서의 데이터가 무시됨
