---
layout : posts
author : Jiyeon Lee
title : "[지연/2nd meetup] Extended ML 키워드 조사"
published : true
---


# 머신러닝 이론

## 회귀와 분류의 공통점과 차이점

> 가장 큰 차이점은 연속성! 공통점은 아무래도 예측하는 것?!

1. 회귀 (Regression)
연속적인 숫자, 또는 부동소수점수 (실수)를 예측하는 것. 주식 가격을 예측하여 수익을 내는 알고 트레이딩 등이 이에 속한다. 

2. 분류 (Classification)
미리 정의된, 가능성 있는 여러 <u>클래스 레이블(class label)</u> 중 하나를 예측하는 것. 얼굴 인식, 숫자 판별 등이 이에 속한다.
- 이진 분류 (binary classification) : 두개로만 나뉨. 한 클래스를 양성(positive), 다른 하나를 음성(negative) 클래스라고 한다. 
- 다중 분류 (multiclass classifiaction) : 셋 이상의 클래스로 분류

---

## 선형 회귀(Linear Regression)란?
어떤 변수에 다른 변수들이 주는 영향력을 <u>선형적으로 분석</u>하는 대표적인 방법.

### 선형 회귀 분석을 하는 방법
1. 선형 회귀 모델 (linear regression model) 만들기
여기서 말하는 모델은 수학 식으로 표현되는 함수를 말한다. 영향을 주는 변수와 영향을 받는 변수로 구성되어 있다. 영향을 주는 변수는 <u>독립 변수(independent variable) 또는 설명 변수(explanatory variable)</u> 등으로 불리며 영향을 받는 변수는 <u>종속 변수(dependent variable) 또는 반응 변수(response variable)</u> 등으로 불린다.

### 독립 변수의 개수에 따른 선형 회귀 모델
1. 독립 변수가 1개일 때 - 단순 선형 회귀 모델 (Simple Linear Regression)
 Y=β0+β1X+ϵ

2. 독립 변수가 n개일 때 - 다중 선형 회귀 모델 (Multiple Linear Regression)
 Y=β0+β1X1+…+βnXn+ϵ

### 최적의 계수를 찾는 방법 - LSE
> LSE (Least Square Estimation)
error 제곱이 최소화가 되도록 계수를 찾는 방법!


---
## 정규화의 방법
정규화는 모델 복잡도에 대한 패널티(penalty)
정규화는 과적합을 예방하고 일반화 성능을 높이는 데 도움을 준다

![정규화](https://t1.daumcdn.net/cfile/tistory/99B7603359820B9228)
![정규화2](https://user-images.githubusercontent.com/26396102/46059078-ae675f00-c198-11e8-9ca7-7aeb9bf0ea3d.PNG)

### 1. L1 Regularization
정규화의 일종. 모델 가중치의 L1 norm(가중치 각 요소 절대값의 합)에 대해 패널티를 부과한다. 대부분의 요소값이 0인 sparse feature에 의존한 모델에서 L1 정규화는 불필요한 피처에 대응하는 가중치들을 정확히 0으로 만들어 해당 피처를 모델이 무시하도록 만든다. 다시 말해변수 선택(feauture selection) 효과가 있다.

### 2. L2 Regularization
정규화의 일종. 모델 가중치의 L2 norm의 제곱(가중치 각 요소 제곱의 합)에 대해 패널티를 부과한다. L2 정규화는 아주 큰 값이나 작은 값을 가지는 outlier 모델 가중치에 대해 0에는 가깝지만 0은 아닌 값으로 만든다. L2 정규화는 선형 모델의 일반화 능력을 언제나 항상 개선시킨다. 

---
## One-hot Encoding, Multi-hot Encoding

### 원-핫 인코딩(One-hot Encoding)이란?
원-핫 인코딩은 단어 집합의 크기를 벡터 차원으로 하고, 표현하고 싶은 단어의 인덱스에 1의 값을 부여하고 다른 인덱스에는 0을 부여하는 단어의 벡터 표현 방식이다. 이렇게 표현된 벡터를 원-핫 벡터(One-hot Vector)라고 한다. 

```python
#원-핫 인코딩 예시
[[0. 0. 1. 0. 0. 0. 0. 0.] #인덱스 2의 원-핫 벡터
 [0. 0. 0. 0. 0. 1. 0. 0.] #인덱스 5의 원-핫 벡터
 [0. 1. 0. 0. 0. 0. 0. 0.] #인덱스 1의 원-핫 벡터
 [0. 0. 0. 0. 0. 0. 1. 0.] #인덱스 6의 원-핫 벡터
 [0. 0. 0. 1. 0. 0. 0. 0.] #인덱스 3의 원-핫 벡터
 [0. 0. 0. 0. 0. 0. 0. 1.]] #인덱스 7의 원-핫 벡터
```
---
## 활성 함수(activation function)
활성함수란, 네트워크에 비선형성(nonlinearlity)을 추가하기 위해 사용됨
- 활성함수 없이 layer를 쌓은 네트워크는 1-layer 네트워크와 동일하기 때문에 활성함수는 비선형 함수로 불리기도 한다.

### 1. 시그모이드 함수 (Sigmoid Function)
![시그모이드](https://t1.daumcdn.net/cfile/tistory/99B9A3335981FCAC0F)
![시그모이드2](https://t1.daumcdn.net/cfile/tistory/995EF8335981FCAC23)

- 결과값이 [0, 1]사이로 제한됨
- 뇌의 유런과 유사하여 많이 쓰였음.
- 문제점
    1. 그라이던트가 죽는 현상이 발생한다. (Gradient Vanishing문제)
         즉, 학습이 되질 않는다.
    2. 활성 함수의 결과 값의 중심이 0이 아닌 0.5다
    3. 계산이 복잡하다 (지수함수 계산)

### 2. ReLU (Rectified Linear Unit)
![Relu2](https://t1.daumcdn.net/cfile/tistory/990A6A335981FFA437)

- 최근 뉴럴 네트워크에서 가장 많이 쓰이는 함수
- 0에서 확 꺾이기 때문에 비선형!
- 장점
    1. 양 극단값이 포화되지 않는다. (양수 지역은 선형적)
    2. 계산이 매우 효율적이다. (최대값 연산 1개)
    3. 수렴 속도가 시그모이드류 함수 대비 6배정도 빠르다. 
- 단점
    1. 중심값이 0이 아님 (마이너한 문제)
    2. 입력 값이 음수인 경우 항상 0을 출력함. (마찬가지로 파라미터 업데이트가 안됨)


---

# NN
## 신경망이란?
신경망은 생물학 모델을 바탕으로한  컴퓨팅의 한 형태로서, layer 로 조직된 많은수의 처리 요소로 구성된 수학 모델, 외부입력에 반응하여 동적으로 정보를 처리하는 많은 간단하지만 고도로 상호 연결된 처리요소로 구성된 컴퓨터 시스템 등으로 정의된다.

## NN의 구조
![nn](https://tb.kibo.or.kr/rbs/modules/techEvaluation/design/default/images/sub02_01.png)

---
# 뉴럴 네트워크 코드 짜는 법
## 1단계 : 전향 전파법 (forward propagation)

![인공뉴런](https://d262ilb51hltx0.cloudfront.net/max/1600/1*ya95fCXH4H7zys8GsrZvng.png)
> 뉴런은 함수와 같다. 몇 가지 입력값을 넣으면 촤라락 계산해서 결과 값이 나온다.

위 사진은 인공 뉴런을 설명합니다. 왼쪽에 보이는 값은 두 입력값과 바이어스(bias) 값이 더해진 것입니다. 바이어스 값이 -2로 설정되어 있는 동안 입력값은 1 또는 0입니다.

두 입력은 7과 3이라는 weight라 불리는 값들로 곱해집니다. 

최종적으로 이 값들을 바이어스 값과 함께 더한 후 5라는 결과값을 내놓습니다. 이게 인공 뉴런의 입력값입니다. 
![인공뉴런2](https://d262ilb51hltx0.cloudfront.net/max/1600/1*PA-u0C_K9LPMgya696Rq4w.png)

뉴런은 이 숫자들을 가지고 어떤 계산을 합니다. 지금 이 경우에는 5를 넣고 계산한 시그모이드 값은 반올림 하면 1이 나옵니다.

![NN](https://d262ilb51hltx0.cloudfront.net/max/1200/1*5GSpUs2hWFx4Lq2_KCyulg.png)

이 뉴런들을 네트어ㅜ크를 통해 연결한다면 전향적인(forward) 뉴런 네트워크를 가질 수 있습니다. 입력값부터 결과까지 시냅스를 통해 각각이 연결되면 위의 이미지처럼 연결된 뉴런이 된다. 

<https://www.youtube.com/watch?v=bxe2T-V8XRs>

## 2단계 : 시그모이드 함수 이해하기
![시그모이드함수](https://d262ilb51hltx0.cloudfront.net/max/1200/1*wjx8PUC97THg7Qw_8qIgEw.png)


## 3단계 : 역 전파법 (Backpropagation)
입력값에서 결과값이 나오기까지 뉴럴 네트워크가 동작하는 과정을 하는 것은 그닥 어렵지 않으나, 실제 데이터 샘플들을 가지고 뉴럴 네트워크가 어떻게 배우는지 이해하는 것이 좀 더 어렵다.
이 개념이 바로 역전파법! (back propagation)이다.

> 기실 의미하는 바는 네트워크가 예측한 것에 비해 얼마나 잘못된 것인지고, 그에 따라 얼마만큼 네트워크 weight를 조정하는 것에 달려있다.

이 과정은 거꾸로 진행된다. 네트워크가 끝나는 부분에서 시작된다. 이 과정은 네트워크가 추축하는 것이 얼마나 잘못된 것인 지를 확인하는 것이다. 

![backpropagation](https://d262ilb51hltx0.cloudfront.net/max/1600/1*cywgo_I0fAPw4QqGh8gwRg.png)

입력값에 도달하기까지 weight를 조정하면서 네트워크에서 반대방향으로 진행된다. 


## 간단한 코드

```python
import numpy as np

#sigmoid function
def nonlin(x, deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

#input dataset
X = np.array([0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1])

# 결과 데이터 값
y = np.array([0, 0, 1, 1]).T 

# 계산을 위한 시드 설정
# 실험의 편의를 위해 항상 같은 값이 나오게 한다.
np.random.seed(1)

# weights를 랜덤적으로 mean of 0으로 초기화하자.
syn0 = 2*np.random.random((3,1)) -1

for iter in range(10000):
    #forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))

    # 우리가 얼마나 놓쳤는지?
    l1_error = y - l1

    # 우리가 놓친 것과 l1의 시모이드 경사와 곱하기
    l1_delta = l1_error * nonlin(l1, True)

    # Weight 업뎃
    syn0 += np.dot(l0.T, l1_delta)

print("Output After Traning")
print(l1)

```

|변수 | 변수에 관한 설명|
|:----:|:----|
| X |	각각의 행들이 트레이닝 샘플인 입력 데이터 행 |
y |	각각의 행들이 트레이닝 샘플인 결과 데이터 행
l0 | 네트워크의 첫 번째 층. 입력 데이터값들을 특징화한다.
l1 | 네트워크의 두 번째 층. 보통 히든 레이어으로 알려져 있다.
syn0 |	weight들의 첫번째 레이어인 시냅스 0로 l0과 l1를 연결시킨다.
|* |	벡터의 원소별 곱셈(Elementwise multiplication). 같은 크기의 두 벡터의 상응하는 값들을 대응시켜 곱한 같은 크기의 벡터를 반환한다.
| - |	벡터의 원소별 뺄셈(Elementwise subtraction). 같은 크기의 두 벡터의 상응하는 값들을 대응시켜 뺀 벡터를 반환한다.
| x.dot(y) | x, y가 벡터라면, 벡터의 내적(dot product)이다. 둘 다 행라면, 행의 곱이고, 만약 오직 하나만이 행라면, 벡터-행 곱이다. |

<https://ddanggle.github.io/11lines>

---

# Reference
- <https://ddanggle.github.io/LearningHowToCodeNeuralNetworks>
