---
layout: default
title: Gradient Descent
parent: Supervised-learning
grand_parent: LG Aimers
use_math: true
nav_order: 1
---


# Gradient Descent

Gradient Descent는 ML, DL에서 parameter optimization을 위해 널리 쓰이는 기법이다.

단순한 방식의 Gradient Descent부터 기본적인 Gradient Descent의 약점을 보완해 만들어진 여러가지 방법이 있는데 이에 대해 요약해보았다.

각각의 방법에 대한 개념적인 부분 및 scikit-learn, PyTorch에서 어떻게 사용할 수 있는지에 대해서도 정리하였다.

[나중에 읽어볼 관련논문: An overview of gradient descent optimization algorithms](https://arxiv.org/abs/1609.04747)

## BGD, SGD, MSGD

기본적인 Gradient Descent 방법이라 이 글에서는 추가적인 정리는 하지 않음.
sklearn에서 SGD를 SGDClassifier, SGDRegressor 등으로 이용할 수 있다(나머지 방식은 패키지로 지원하지는 않아 필요하다면 직접 구현해야 함)
pytorch에서는 BGD, SGD, MSGD 모두 구현이 가능하다(sklearn 느낌의 완성품 패키지는 없으며 torch.optim.SGD를 적절하게 사용하면 BGD, SGD, MSGD 모두 구현이 가능하다.  

## Momentum을 추가한 방식
### Momentum이란?
Momentum의 의미를 조사해보면 물리학적으로 "물체가 한 방향으로 지속적으로 변동하려는 경향"이라는 의미가 있으며 주식에서는 주가의 상승 및 하락의 기세를 의미하는 용어로 사용된다. 정리해보면, Momentum은 축적된 기세, 방향을 의미하는 것이다.

위 문단에서 알아보았듯이, Gradient Descent에서도 Momentum은 유사한 의미로 활용된다. 바로 "이때까지 이동해온 gradient의 축적된 정보"로 말이다. 그렇다면 Momentum을 굳이 왜 추가하는 것일까?

### Motivation
Gradient descent는 좋은 방법이지만 약점도 존재한다. 바로 local optimum에 빠져서 global optimum을 찾아갈 수 없을 가능성이다. Momentum은 gradient descent과정에서 local optimum을 더 잘 빠져나가기 위해서 사용하는 것이다. 원리는 다음과 같다.

Gradient descent를 통해 local optimum에 도달했다고 가정하자. momentum을 사용하지 않은 GD의 경우에 gradient가 0이 되버리며 학습이 종료된다. 그러나 momentum 정보를 추가한 상태라면 이전에 쌓여왔던 정보를 이용해 local optimum을 벗어날 수도 있는 것이다.

### SGD + Momentum
$\mathbf{v}\_t = \rho \mathbf{v}\_{t-1} + \alpha \nabla J(\mathbf{\theta}\_t)$

$\mathbf{\theta}\_{t} = \mathbf{\theta}\_{t-1} - \mathbf{v}\_{t}$

$\rho$는 momentum term으로써 0 ~ 1의 범위를 가지며 보통 0.9 또는 유사한 값으로 설정된다고 한다. 즉 parameter update 시에 현재 시점의 gradient 뿐 아니라 이전 스텝의 gradient 정보도 반영해주는 방식이다.

### Nesterov Accelerated Gradient (NAG)
momentum을 사용하지 않는 경우 우리의 다음 step은 온전히 현재의 gradient에 달려 있어 어디로 이동할 지 예측하기 어렵다. 하지만 momentum 정보를 이용하면 다음 step에 대해 대략적인 예상을 할 수 있게 된다($\mathbf{\theta}\_{t-1} - \rho \mathbf{v}\_{t-1}$이 다음 parameter와 유사한 값이기 때문). 

$\mathbf{v}\_t = \rho \mathbf{v}\_{t-1} + \alpha \nabla J(\mathbf{\theta}\_{t-1} - \rho \mathbf{v}\_{t-1})$

$\mathbf{\theta}\_t = \mathbf{\theta}\_{t-1} - \mathbf{v}\_t$

NAG를 사용하면 이전의 모멘텀을 기반으로 미래를 예측해 파라미터를 업데이트하므로 SGD보다 더 안정적이고 빠르게 수렴할 수 있다. 이러한 장점이 잘 활용되어 다양한 RNN task 등등에 널리 이용되고 있다.

### torch.optim.SGD로 구현하기
sklearn에서는 위의 방법들(BGD, SGD, MSGD, Momentum, NAG) 중 SGD를 제외하고는 패키지로 제공되는 것이 없다. 따라서 torch.optim.SGD를 이용해서 Momentum, NAG를 사용하는 방법에 대해 정리해보았다.
```
import torch

model = ...  # 적절한 모델이 있다고 가정
# NAG를 사용하는 optimizer 정의하기
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,  # learning rate
    momentum=0.9,  # momentum factor (default는 0)
    dampening=0,  # velocity에 현재의 gradient를 얼마나 반영시키는 지 결정 (default는 0)
    weight_decay=0,  # L2-regularization을 사용하는 경우 regularization term의 값
    nesterov=True,  # NAG를 사용할 지 여부 (default는 False) 
)
```

## Per-parameter learning rate를 추가한 방식
feature마다 scale이 다를 수도 있고 서로 최적화되는 시점도 다를 수 있으므로 feature 별로 learning rate를 다르게 해주는 것이 효과적일 것이라는 아이디어를 적용한 방법.

### AdaGrad (Adaptive Gradient)
$\mathbf{\theta}$가 d차원이라고 한다면 $i \in [1, d]$에 대해서

$\mathbf{\theta}\_{t, i} = \mathbf{\theta}\_{t-1, i} - \frac{\alpha}{\sqrt{\mathbf{G}\_{t, ii} + \epsilon}} \nabla J(\mathbf{\theta}\_{t, i})$

- $\mathbf{G}\_{t}$는 diagonal matrix이며 각각의 diagonal entry $\mathbf{G}\_{ii}$는 $\mathbf{\theta}\_{i}$의 time 1~t까지의 gradient의 squared sum.
- $\epsilon$은 division by zero를 방지하기 위한 것으로 1e-8 정도의 값을 사용함
- squared root를 취한 버전이 더 잘 작동한다고 함

#### AdaGrad의 장점
- parameter별 learning rate가 automatic tuning이 된다
	- gradient 제곱의 합이 커질수록 학습률이 작아지기 때문
	- feature 별로 gradient를 관리할 수 있음

#### AdaGrad의 단점
- gradient의 제곱을 계속 더하는 과정으로 인해 나중에는 learning rate가 너무 작아져서 학습이 일어나지 않음

#### PyTorch에서 AdaGrad 사용하기
PyTorch AdaGrad 알고리즘은 learning rate scheduling이 적용되어있고 epsilon을 루트 외부에 더하여서 위와 계산식이 약간 다름.
```
import torch

model = ...  # 적절한 모델이 있다고 가정
optimizer = torch.optim.Adagrad(
    model.parameters(),
    lr=0.01,  # default가 0.01
    lr_decay=0,  # learning rate scheduling에 사용됨(0이면 schedule 안함)
    weight_decay=0,  # L2-regularization 사용 시에 적용
    initial_accumulator_value=0,
    eps=1e-10,  # division by zero 방지 (default는 1e-10)
)
```

### RMSProp
AdaGrad의 단점을 해결하기 위한 방법. gradient의 squared sum을 더하는 방식을 변경함

$E[g^2]\_t = \rho E[g^2]\_{t-1} + (1-\rho)g\_t^2$
- $\mathbf{G}\_{t, ii}$ 대신에 $E[g^2]\_t$를 사용함.

#### PyTorch에서 RMSProp 사용하기
```
import torch

model = ...  # 적절한 모델이 있다고 가정
optimizer = torch.optim.RMSProp(
    model.parameters(),
    lr=0.01,
    alpha=0.99,
    eps=1e-8,
    weight_decay=0,
    momentum=0,
    centered=False
)
```

### Adam (Adaptive Moment Estimation)
AdaGrad와 RMSProp을 합친 버전
- RMSProp와 유사하게 이전 gradient들의 squared sum을 저장함
- AdaGrad와 유사하게 이전 gradient들을 momentum 형태로 저장함

$m\_t = \beta\_1 m\_{t-1} + (1-\beta\_1)g\_t$
- mean
- $\beta\_1$은 보통 0.9
$m\_t = \frac{m\_t}{1-\beta\_1^t}$
- $m\_t$는 0으로 초기화되기 때문에 초기 과정에서 0으로 치우치는 것을 발견함. 이러한 bias를 없애기 위해 나눠주는 것.

$v\_t = \beta\_2 v\_{t-1} + (1-\beta\_2)g\_t^2$
- variance
- $\beta_2$는 보통 0.999
$v\_t = \frac{v\_t}{1 - \beta\_2^t}$

$\theta\_{t, i} = \theta\_{t-1, i} - \alpha \frac{s_t}{\sqrt{r\_t + \epsilon}}$
- $\epsilon$은 1e-8 추천

#### PyTorch에서 Adam 사용하기
```
import torch

model = ...  # 적절한 모델이 있다고 가정
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0,
    amsgrad=False  # Adam에서 조금 변형된 버전이라고 함
)
```

**대부분의 케이스에서 가장 좋은 선택은 Adam이라고 함**
