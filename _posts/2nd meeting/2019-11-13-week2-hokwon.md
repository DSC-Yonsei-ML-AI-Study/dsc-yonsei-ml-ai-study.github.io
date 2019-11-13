---
layout : posts
author : Hokwon Lee
title : "[호권/2nd meetup] Extended ML 키워드 조사"
published : true
---

# 이번 회차 Keyword
1. 머신러닝 이론
    - 회귀와 분류의 공통점과 차이점
     □ 분류(Classification)
      - 미리 정의된, 가능성 있는 여러 class label중 하나를 예측하는 것!
        ◇ 이진 분류(binary classfication)
          - 두개로만 나누어 분류.
          - ex) 예/아니오로 구분되는 문제
        ◇ 다증 분류(multi-class classification)
          - 셋 이상의 클래스로 분류.
          - ex) 언어 분류 모델(영어 / 프랑스어 / 독일어/ 이탈리아어 / 스페인어 / 한국어...)

      □ 회귀 (Regression)
      - 보통 연속적인 숫자 중 하나를 예측하는데 사용 (연속성)
      - ex) 집의 크기에 따른 매매가격, 공부시간에 따른 전공 시험 점수 예측

    - 선형 회귀란?
      Linear Regression(선형 회귀) 종속 변수 y 와 한 개 이상의 독립 변수
      x와의 선형 상관 관계를 모델링하는 회귀분석 기법
      (주어진 데이터가 어떤 함수로부터 생성됐는가를 알아보는 '함수 관계'를 추측)
      □ 단순 선형 회귀
       - 한개의 설명 변수에 기반하는 경우
      □ 다중 선형 회귀
       - 둘 이상의 설명 변수에 기반하는 경우

    - 손실에 대하여 (L2손실, L1손실 등)
      □ L2 손실
       - 선형회귀에 쓰이는 손실 함수
       - 모델이 예측한 값과 실제값 간 차이의 제곱
       - 장점 : 미분이 가능하다.
       - 단점 : 오차를 제곱하기 때문에 잘못된 예측 혹은 이상치(outlier)에 의해
                그 값이 큰 영향을 받게 된다.
      □ L1 손실
       - 모델이 예측한 값과 실제값 간 차이의 절대값
       - 장점 : L2 손실에 비해 이상치(outlier)에 덜 민감
       - 단점 : 0인 지점에서 미분이 불가능

      □ L1손실, L2 손실의 차이
       - L1 손실이 L2 손실에 비해 이상치(outlier)에 대하여 덜 민감
       - 이상치(outlier)가 적당히 무시되길 원할때 L1 손실 사용, 이상치(outlier)를
         신경써야 하는 경우 L2 손실 사용

    - L2 와 L1의 정규화. 그리고 이 둘의 비교
      □ 정규화
       - 일반화라고 이해하면 편함
       - 모델 복잡도에 대한 패널티로 정규화는 Overfitting을 예방하고 Generalization(일반화)
         성능을 높이는데 도움을 준다.

      □ L1 정규화
       - 수식 :
       ![L1_Regularization](https://user-images.githubusercontent.com/53327933/68753800-f40b1380-0648-11ea-95d8-ef7897ed2ef2.png)
       - cost function에 가중치의 절대값을 더해준다.
       - 기존의 cost function에 가중치의 크기가 포함되면서 가중치가 너무 크지 않는
         방향으로 학습 되도록 한다.
       - λ는 learning rate(학습률) 같은 상수로 0에 가까울 수록 정규화의 효과는 없어진다.
       - LASSOR(Least Absolute Shrinkage and Selection Operater(Lasso) Regression)
         L1 정규화를 사용하는 정규화 모델

      □ L2 정규화
       - 수식 :
       ![L2_Regularization](https://user-images.githubusercontent.com/53327933/68754103-7693d300-0649-11ea-903e-6b1fee75bd0a.png)
       - cost function에 가중치의 제곱을 포함하여 더해준다.
       - L1 정규화와 마찬가지로 가중치가 너무 크지 않는 방향으로 학습
       - Weight decay라고도 불림
       - Ridge Regression : L2 정규화를 사용하는 정규화 모델

      □ L1 정규화, L2 정규화의 차이
       - 가중치 w 가 작아지도록 학습한 다는 것은 결국 Local noise에 영향을 덜 받도록
         하겠다는 것이며 Outlier의 영향을 더 적게 받도록 하겠다는 것.
       - L2 정규화의 경우 각각의 값에 대해 항상 Unique 한 값을 냄
         L1 정규화의 경우 특정 Feature 없이도 같은 값을 낼 수 있다.
       - L2 정규화는 모두 미분 가능
         L1 정규화는 미분 불가능한 점이 있다.


    - 손실을 줄이는 방법 - 경사하강법과 학습률에 대하여
      □ 반복방식
       - 주로 대규모 데이터 세트에 적용하기 용이
       - 하나 이상의 특성을 입력하여 하나의 예측을 출력한다.
       - 손실 값이 가장 낮은 모델 매개변수를 발견할 때 까지 반복 학습
         (전체 손실이 변하지 않거나 매우 느리게 변할 때까지 계속 반복)

      □ 경사하강법
       - 1차 근삿값 발견용 최적화 알고리즘
       - 함수의 기울기를 구하여 기울기가 낮은 쪽으로 계속 이동시켜서 극값에 이를 때까지
         반복시키는 것.
       - 수식 : Xi+1 = Xi - ri▽f(xi) (ri는 이동할 거리를 조절하는 매개 변수)
       - 장점 : 모든 차원과 모든 공간에서 적용이 가능
       - 단점 : 정확성을 위해서 극값으로 이동함에 있어 매우 많은 단계를 거쳐야 함
                주어진 함수에서의 곡률에 따라서 거의 같은 위치에서 시작했음에도
                완전히 다른 결과로 이어질 수도 있다.

      □ 학습률
       - 한 번 학습할 때 얼마만큼 학습해야 하는지의 양(경사하강법의 이동할 거리 ri)
       - a(t) = a(0)(1.0 - t/rlen)
       - 학습률 값은 미리 특정 값을 정해 두어야 한다.
       - 학습률이 너무 크면 큰 값을 반환, 너무 작으면 거의 갱신되지 않고 학습이 끝남
         (너무 크거나 너무 작으면 적합한 지점에 찾아가기 어렵다.)
       - 하이퍼파라미터(hyperparameter), 사람이 수동적으로 설정 해야 함.

    - 과적합이란?
     - ML에서 학습 데이터를 과하게 학습(overfitting)하는 것
     - 학습 데이터에 대해서는 오차가 감소하지만, 실제 데이터에 대해서는 오차가 증가하는
       지점이 존재할 수 있다.
     - Overfitting은 ML 알고리즘의 오차를 증가시키는 원인
        ◇ Overfitting을 해결하기 어렵거나 불가능한 이유
         - 일반적으로 학습 데이터는 실제 데이터의 부분집합, 실제 데이터를 모두 수집하는
           것은 불가능하다.
         - 만약 실제 데이터를 모두 수집하여도 모든 데이터를 학습시키기 위한 시간이 측정
           불가능한 수준으로 증가할 수 있다.
         - 학습 데이터만 가지고 실제 데이터의 오차가 증가하는 지점을 예측하는 것은
           매우 어렵거나 불가능 하다.
     □ Overfitting 해결 방법
      ◇ 최적화 (Optimization)
       - SVM(Support Vector Machine)을 이용
       - SVM은 데이터를 분류하기 위한 decision surface를 찾는 것과 동시에
         각 데이터의 집합과 decision surface간의 거리(margin) 최대화하는 방식으로 학습
       - Deep Learning 이전의 가장 뛰어난 알고리즘

    - 데이터셋을 분할하는 방법 (학습, 검증, 테스트)
     □ 과적합(Overfitting)을 방지하귀 위한 분할
      - 학습데이터(training data) : 모형 f를 추정하는데 필요함
      - 검증데이터(validation data) : 추정한 모형 f가 적합한지 검증
      - 테스트데이터(testing data) : 최정적으로 선택한 모형의 성능을 평가
      - 학습데이터, 검증데이터, 테스트데이터 비율 - 5:3:2

    - 특징 벡터, 특징 추출이란
     □ 특징 벡터(Feature Vector)
      - 피쳐(Feature) : 관습 대상에게서 발견된 개별적이고 측정가능한 경험적 속성
      - 특징 벡터는 Feature Vector로 피쳐들의 집합이다.
      - 수학적으로 다루기 편리하여 Vector로 표현
     □ 특징 추출 (Feature Extaction)
      - 고차원의 원본 피쳐 공간을 저차원의 새로운 피쳐 공간으로 투영시킨다.
        새롭게 구성된 피쳐 공간은 보통 원본 피쳐 공간의 선형 또는 비선형 결합.

    - OOV(Out Of Vocabulary)
     - 단어셋에 없는 단어들의 집합

    - 원 핫 인코딩, 멀티 핫 인코딩
     □ 원-핫 인코딩 (one-hot encoding)
      - 특징 : 요소 중 하나가 1(Hot, True)로 설정
               다른 요소는 모두 0(Cold, False)으로 설정
      - 가능한 값의 유한집합을 갖는 문자열 또는 식별자를 표현하는 데 널리 사용
    □ 멀티-핫 인코딩 (Multi-hot encoding)
     -????????????????????????????????

    - 선형 문제와 비선형 문제
     □ 선형 문제
      - 선형 조건들을 만족시키면서 선형인 목적 함수를 최적화 하는 문제.
     □ 비선형 문제
      - 목적 함수의 제약 조건 중 일부가 비선형인 최적화 문제.
      - 미상의 실수형 변수 집합에서 손실 함수의 극값의 계산의 하나이며 총괄하여
        제약 조건으로 불리는 등식과 부등식의 체계의 만족에 조건적

    - 시그모이드 함수, ReLU 함수의 특징과 적용
     □ 시그모이드(sigmoid) 함수
      - 수식 : s(z) = 1 / (1 + e^-z)
      - 1과 0 사이의 값을 가진다.
      - 모든 실수 입력 값이 미분 가능하다.
      - 로지스틱 회귀 분석에 적용.
      - 단점
        1. saturation현상 발생
        2. squarshing을 하려고 하지 않아도 값이 0~1만 나옴, ZigZag현상 발생.
        3. exp연산이 들어가므로 연산이 느리다.
     □ ReLu 함수
      - sigmoid의 단점을 보안한 activation 함수
      - 딥러닝 activation 함수의 기본
      - 수식 : f(x) - max(0,x)
      - 장점 : 구조가 단순하고 속도가 빠르다.
      - 단점 : sigmoid처럼 zigzag현상이 생긴다.

    - 로지스틱 회귀란? 로지스틱 회귀 모델에서의 정규화, 손실, 임계값
     □ 로지스틱 회귀
      - 분류 문제에서 선형 예측에 sigmoid 함수를 적용하여 가능한 각 불연속 라벨값에
        대한 확률을 생성하는 모델
      - 이진 분류 문제에 흔히 사용, 다중 클래스 분류 문제에도 사용 가능
      ◇ 정규화
       - L2정규화 이용
       - 조기 중단, 즉 학습 단계 수 또는 학습률을 제한.
       ![Regularization_Logistic_Regression](https://user-images.githubusercontent.com/53327933/68754207-9deaa000-0649-11ea-98df-2f804c14270a.PNG)
      ◇ 손실
       - 로그 손실
       - 수식 :
       ![Log_loss](https://user-images.githubusercontent.com/53327933/68754234-a9d66200-0649-11ea-99d3-755fcf02c402.PNG)
      ◇ 임계값
       - 애매한 값을 이분법으로 분류 할 기준
       - 로지스틱 회귀 값을 이진 카테고리에 매핑 하려면 분류 임계값을 정의해야 한다.
       - 스팸 : 임계값보다 높은 값
         낫스팸 : 임계값보다 낮은 값

2. NN
    - 신경망이란?
     - 인공신경망은 생물학의 신경망에서 영감을 얻은 학습 알고리즘, 시냅스의 결합으로
       네트워크를 형성한 인공 뉴런이 학습을 통해 시냅스의 결합 세기를 변화시켜 문제해결
       능력을 가지는 비선형 모델

    - NN의 구조 (어떻게 구성되는가?)
     ![NN의 구조](https://user-images.githubusercontent.com/53327933/68754260-b5c22400-0649-11ea-8dc2-b7e1263d6f65.PNG)

    - 전향 전파법 (forward propagation)
     - 뉴럴 네트워크 모델의 입력층부터 출력층까지 순서대로 변수들을 계산하고 저장하는 것
       간단하게 하기 위해서, 입력은  d  차원의 실수 공간  x∈Rd  으로 부터 선택되고,
       편향(bias) 항목은 생략하겠습니다. 중간 변수는 다음과 같이 정의됩니다.

                              z=W(1)x

       W(1)∈Rh×d  은 은닉층(hidden layer)의 가중치 파라미터입니다.
       중간 변수  z∈Rh  를 활성화 함수(activation functino)  ϕ  에 입력해서
       벡터 길이가  h  인 은닉층(hidden layer) 변수를 얻습니다.

                              h=ϕ(z).

       은닉 변수  h  도 중간 변수입니다.
       출력층의 가중치  W(2)∈Rq×h  만을 사용한다고 가정하면,
       벡터 길이가  q  인 출력층의 변수를 다음과 같이 계산할 수 있습니다.

                            o=W(2)h.

       손실 함수(loss function)를  l  이라고 하고,
       샘플 레이블을  y  라고 가정하면,
       하나의 데이터 샘플에 대한 손실(loss) 값을 다음과 같이 계산할 수 있습니다.

                            L=l(o,y).

       ℓ2  놈 정규화(norm regularization)의 정의에 따라서,
       하이퍼파라미터(hyper-parameter)  λ  가 주어졌을 때,
       정규화 (regularization) 항목은 다음과 같습니다.

                    s=λ2(∥W(1)∥2F+∥W(2)∥2F).

       여기서 행렬의 Frobenius norm은 행렬을 벡터로 바꾼 후 계산하는
       L2  놈(norm)과 같습니다. 마지막으로, 한개의 데이터 샘플에 대한 모델의
       정규화된 손실(regularized loss) 값을 계산합니다.

                            J=L+s.

       J  를 주어진 데이터 샘플에 대한 목표 함수(objective function)라고 하며,
       앞으로 이를 '목표 함수(objective function)'라고 하겠습니다.

    - 시그모이드 함수를 어디에 적용하는가?
      - 활성 함수로서 로지스틱 회귀에 알고리즘에 이용한다.

    - 역전파법 (Backpropagation)
      - 중간 변수와 파라미터에 대한 그래디언트(gradient)를 반대 방향으로 계산하고 저장합니다.
      일반적으로는 역전파(back propagation)은 뉴럴 네트워크의 각 층과 관련된
      목적 함수(objective function)의 중간 변수들과 파라미터들의 그래디언트(gradient)를
      출력층에서 입력층 순으로 계산하고 저장합니다.
      이는 미적분의 ’체인룰(chain rule)’을 따르기 때문입니다.
      임의의 모양을 갖는 입력과 출력 텐서(tensor)  X,Y,Z  들을 이용해서 함수  Y=f(X)  와
      Z=g(Y)=g∘f(X)  를 정의했다고 가정하고, 체인룰(chain rule)을 사용하면,  
      X  에 대한  Z  의 미분은 다음과 같이 정의됩니다.

                              ∂Z∂X=prod(∂Z∂Y,∂Y∂X).

     여기서  prod  연산은 전치(transposotion)나 입력 위치 변경과 같이 필요한 연산을 수항한 후 곱을 수행하는 것을 의미합니다. 벡터의 경우에는 이것은 직관적입니다. 단순히 행렬-행렬 곱셈이고, 고차원의 텐서의 경우에는 새로 대응하는 원소들 간에 연산을 수행합니다.  prod  연산자는 이 모든 복잡한 개념을 감춰주는 역할을 합니다.

     하나의 은닉층(hidden layer)를 갖는 간단한 네트워크의 파라매터는  W(1)  와  W(2)  이고, 역전파(back propagation)는 미분값  ∂J/∂W(1)  와  ∂J/∂W(2)  를 계산하는 것입니다. 이를 위해서 우리는 체인룰(chain rule)을 적용해서 각 중간 변수와 파라미터에 대한 그래디언트(gradient)를 계산합니다. 연산 그래프의 결과로부터 시작해서 파라미터들에 대한 그래디언트(gradient)를 계산해야하기 때문에, 순전파(forward propagation)와는 반대 방향으로 연산을 수행합니다. 첫번째 단계는 손실(loss) 항목  L  과 정규화(regularization) 항목  s  에 대해서 목적 함수(objective function)  J=L+s  의 그래디언트(gradient)를 계산하는 것입니다.

                              ∂J∂L=1 and ∂J∂s=1

     그 다음, 출력층  o  의 변수들에 대한 목적 함수(objective function)의 그래디언트(gradient)를 체인룰(chain rule)을 적용해서 구합니다.

                        ∂J∂o=prod(∂J∂L,∂L∂o)=∂L∂o∈Rq

     이제 두 파라메터에 대해서 정규화(regularization) 항목의 그래디언트(gradient)를 계산합니다.

                      ∂s∂W(1)=λW(1) and ∂s∂W(2)=λW(2)

     이제 우리는 출력층와 가장 가까운 모델 파라미터들에 대해서 목적 함수(objective function)의 그래디언트(gradient)  ∂J/∂W(2)∈Rq×h  를 계산할 수 있습니다. 체인룰(chain rule)을 적용하면 다음과 같이 계산됩니다.

            ∂J∂W(2)=prod(∂J∂o,∂o∂W(2))+prod(∂J∂s,∂s∂W(2))=∂J∂oh⊤+λW(2)

     W(1)  에 대한 그래디언트(gradient)를 계산하기 위해서, 출력층으로부터 은닉층까지 역전파(back propagation)를 계속 해야합니다. 은닉층(hidden layer) 변수에 대한 그래디언트(gradient)  ∂J/∂h∈Rh  는 다음과 같습니다.

                    ∂J∂h=prod(∂J∂o,∂o∂h)=W(2)⊤∂J∂o.

     활성화 함수(activation function)  ϕ  는 각 요소별로 적용되기 때문에, 중간 변수  z  에 대한 그래디언트(gradient)  ∂J/∂z∈Rh  를 계산하기 위해서는 요소별 곱하기(element-wise multiplication) 연산자를 사용해야합니다. 우리는 이 연산을  ⊙  로 표현하겠습니다.

                  ∂J∂z=prod(∂J∂h,∂h∂z)=∂J∂h⊙ϕ′(z).

     마지막으로, 입력층과 가장 가까운 모델 파라미터에 대한 그래디언트(gradient)  ∂J/∂W(1)∈Rh×d  를 체인룰(chain rule)을 적용해서 다음과 같이 계산합니다.

        ∂J∂W(1)=prod(∂J∂z,∂z∂W(1))+prod(∂J∂s,∂s∂W(1))=∂J∂zx⊤+λW(1).


-https://light-tree.tistory.com/125 (L1, L2 norm, Regularization, loss 참고)
-https://ko.d2l.ai/chapter_deep-learning-basics/backprop.html
