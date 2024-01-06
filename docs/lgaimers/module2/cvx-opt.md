---
layout: default
title: Convex Optimization
parent: Mathematics for ML
grand_parent: LG Aimers
use_math: true
nav_order: 2
---
# Convex Optimization
## Optimization이 ML에서 중요한 이유
ML모델을 training하는 과정이 보통 최적의 parameter를 찾아내는 문제이다. 즉 Optimization 문제가 되는 경우가 많음.

## Optimization problem의 분류
1. constraints가 있는지에 따라 
- Constrained optimization
- Unconstrained optimzation

2. objective function, constraints가 convex인지에 따라
- Convex optimization 인지 여부로 분류 가능

## Unconstrained optimization
Unconstrained optimization problem인 경우 gradient descent를 바로 활용할 수 있다.

### Gradient Descent의 개념
objective function이 $f(\mathbf{x})$이고 epoch당 step size (learning rate)가 $\gamma_k$일 때,
1. step size의 값이 적절하고
2. 방향 $\mathbf{d}_k$가 $\nabla f(\mathbf{x}_k) \cdot \mathbf{d}_k < 0$

인 경우 local optimum을 구할 수 있음이 보장됨.
- gradient vector는 각 point에서 가장 빠르게 증가하는 방향으로 흐르기 때문에 gradient vector의 반대 방향으로 $\mathbf{d}_k$를 설정
- step마다 learning rate가 달라질 수 있음(learning rate schedule을 사용하는 경우)
	- sklearn의 SGDClassifier에서 learning_rate를 점진적으로 감소시키는 등의 방법

### Gradient Descent의 종류
일반적으로 최적화시켜야 하는 loss function은 각각의 data point에 대한 loss function을 더한 것으로 정의되는 경우가 많다(예를 들면 Linear regression에서 loss function은 MSE를 이용해 정의됨). 따라서 gradient를 구할 때도 각각의 data point에 대한 gradient를 모두 더하여 전체 gradient를 구해야하는 경우가 많다.

#### 1. Batch(Full) Gradient Descent


위에서 설명한 기본적인 방식을 그대로 구현하는 GD 알고리즘.
$\mathbf{\theta}_{k+1} = \mathbf{\theta}_k - \gamma_k \sum^n\_{i=1} \nabla f(\mathbf{x}^{(i)})$
- sklearn에서 제공하지 않음
- 장점
	- 안정적으로 $\theta^*$를 찾아갈 수 있다 
- 단점
	- 안정적인 만큼 local minimum에 빠질 수 있다
	- step 마다 모든 data point에 대한 gradient를 구해야 하므로 오래 걸린다


#### 2. Stochastic Gradient Descent


step 마다 전체 data point 중 하나를 랜덤으로 선택한 뒤 그 data point의 gradient만 사용하여 파라미터를 최적화시키는 방식. data point의 개수를 m이라고 할 때, 한 epoch마다 랜덤으로 data point를 선택하여 파라미터를 업데이트 하는 과정을 m번 반복함.
$\mathbf{\theta}_{k+1} = \mathbf{\theta}_k - \gamma_k \nabla f(\mathbf{x}^{(i)})$
- 장점
	- 한 step당 계산 시간이 줄어드므로 더 빠르게 global optimum에 도달할 수 있음
	- 한번에 하나의 data point만 필요하므로 아주 큰 dataset에도 적용이 가능
- 단점
	- 무작위성으로 인해 global optimum을 벗어날 확률마저 높아짐
- learning rate schedule
	- 위에서 언급한 SGD의 단점을 극복하기 위한 방법
	- learning rate를 점진적으로 감소시켜 global optimum에 안정적으로 도달할 수 있게 해주는 방식
- SGD를 적용하기 위한 조건
	- dataset이 IID condition을 만족해야 함
		- IID condition을 갖추어야만 global optimum을 향해 간다는 것이 보장됨
		- 모든 data point가 서로 독립적이며(영향을 주지 않고) 같은 분포를 가져야 함
		- epoch 시작 또는 끝에 data point는 shuffle하거나 step 마다 data point를 랜덤으로 선택하는 방식을 사용하면 보장됨
	- feature의 scale이 동일해야 함
		- scale이 동일하지 않다면 global optimum에 도달하는 시간이 더 길어짐
		- sklearn의 StandardScaler() 이용하기
- sklearn에서 SGD 사용하기
	- SGDClassifier: default learning rate schedule 방식은 $\gamma_k = \frac{1}{alpha(t_0 + k)}$
	- SGDRegressor: default learning rate schedule 방식은 $\gamma_k = \frac{eta0}{k^{power\_t}}$

```
'''
적절한 dataset X, y가 주어진 경우
'''
import numpy as np
from sklearn.linear_model import SGDRegressor

sgd_reg = SGDRegressor()
sgd_reg.fit(X, y.ravel())
```


#### 3. Mini-batch Gradient Descent


dataset을 mini-batch라고 부르는 작은 sample set으로 나누고 각각의 set에 대해서 FGD를 적용하는 방식.
GPU 최적화를 잘 이용할 수 있다
- sklearn에서는 제공하지 않음