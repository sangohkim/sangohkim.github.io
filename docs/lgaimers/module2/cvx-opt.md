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


## Constrained Optimization
Constrained Optimization인 경우 GD 등의 방법을 사용할 수 있게 unconstrained optimization problem으로 바꾼 후에 풀어야 함
### Standard Constrained Optimization Problem
optimization problem의 standard form:
$minimize f(\mathbf{x})$ subject to 
$g_i(\mathbf{x}) \le 0$ for $i=1,...,m$ (Inequality constraints)
$h_j(\mathbf{x}) = 0$ for $j=1,...,p$ (Equality constraints)

### Problem solving via Lagrange Multipliers
#### Duality Mentality
Primal problem optimal value를 구할 수 있거나 또는 primal optimal value에 대한 bound를 제공할 수 있는 dual problem을 풀어서 primal problem의 optimal value를 구하자

#### Dual Problem 만들기
Lagrangian: $L(\mathbf{x}, \mathbf{\lambda}, \mathbf{\nu}) = f(\mathbf{x}) + \sum^m_{i=1}\lambda_i g_i(\mathbf{x}) + \sum^p_{j=1}\nu_j h_j(\mathbf{x})$
($\mathbf{\lambda}_i \ge 0$)

Lagrangian Dual Function: $D(\mathbf{\lambda}, \mathbf{\nu}) = inf_{\mathbf{x}}L(\mathbf{x}, \mathbf{\lambda}, \mathbf{\nu})$
- $\mathbf{\lambda}, \mathbf{\nu}$가 고정되었을 때 Lagrangian의 최솟값(infimum)
- primal optimal value $p^*$의 lower bound가 된다
	- primal optimal value는 결국 $f(\mathbf{x})$의 값인데 Lagrangian이 항상 $f(\mathbf{x})$보다 작음
	- $D(\mathbf{\lambda}, \mathbf{\nu}) \le p^*$가 항상 성립

Lagrangian Dual Problem:
$max_{\mathbf{\lambda}, \mathbf{\nu}} D(\mathbf{\lambda}, \mathbf{\nu})$ subject to $\mathbf{\lambda} \succeq 0$
- $D(\mathbf{\lambda}, \mathbf{\nu}) \le p^*$가 항상 성립하므로 Lagrangian dual function의 최댓값을 구하면 그 값이 primal optimal value에 대한 best lower bound가 됨
- Lagrangian Dual Problem은 항상 convex optimization
	- GD 등의 방법을 이용해 풀 수 있음이 보장됨

**즉, constrained optimization problem의 경우 Dual Problem으로 바꿔서 푼다**

### Duality
primal problem과 dual problem 사이의 관계를 의미

primal problem optimal value: $p^\*$, dual problem optimal value: $d^\*$라고 하면
#### Weak Duality
$d^* \le p^*$
- primal, dual problem 사이에서 항상 성립하는 성질
- $p^* - d^*$를 optimal duality gap이라고 함

#### Strong Duality
$d^* = p^*$
- optimal duality gap이 0
- 대부분의 convex optimization problem에서 성립함(추가적인 몇가지 조건이 필요)

### Convex Optimization
#### Standard Convex optimization problem
$minimize f(\mathbf{x})$
subject to $g_i(\mathbf{x}) \le 0$ for i = 1, ..., m
subject to $\mathbf{a}_j^T\mathbf{x}=\mathbf{b}_j$ for j = 1, ..., p
(f, g, h 모두 convex이며 정의역도 convex set이어야 함)
- upper bound inequality on "convex" function $\Rightarrow$ Constraint set이 convex
- equality constraint는 affine하므로 convex

#### Convex Set 이란?
- 수학적 정의
	- Set $C$가 $\forall \mathbf{x}, \mathbf{y} \in C, \theta \in [0, 1]$ 일 때 $\theta \mathbf{x} + (1-\theta) \mathbf{y} \in C$가 성립하면 convex set이라고 함
	- 즉, set 내부의 임의의 두 점을 잇는 선분이 set 밖으로 나가지 않으면 convex set
- 직관적으로 볼록한 모양의 set이라고 생각하면 됨

#### Convex function이란?
함수 $f: \mathbb{R}^n \rightarrow \mathbb{R}$의 domain이 convex set이고 domain 에 속하는 임의의 두 $\mathbf{x}, \mathbf{y}$가 $\theta \in [0, 1]$일 때 $f(\theta \mathbf{x} + (1-\theta)\mathbf{y}) \le \theta f(\mathbf{x}) + (1-\theta) f(\mathbf{y})$를 만족하면 f를 convex funtion이라고 한다.
- $0 \lt \theta \lt 1$인 경우 strictly convex라고 함.
- concave: -f가 convex인 경우
- affine function: convex이면서 concave인 function
	- linear function이 대표적인 예시

#### Convex function의 특징
- local minimum이 곧 global minimum이다.
- curvature(곡률)이 항상 증가한다(접선의 기울기가 항상 증가한다)

#### Convex function의 예시
$f(\mathbf{x}) = max\{x_1, x_2, ..., x_n\}$
$f(\mathbf{x}) = log\sum^n_{i=1}e^{x_i}$

#### Convex-preserving operation
1. $f = \sum^n_{i=1}w_if_i$
2. $g(\mathbf{x}) = f(a\mathbf{x} + b)$
3. $f=max\{f_1, f_2\}$
4. 기타 등등

#### Lagrangian Dual Function이 Convex function인 이유
- Point-wise supremum
	- $f(\mathbf{x}, \mathbf{y})$가 y를 고정했을 때 $\mathbf{x}$에 대해서 convex이면 $g(\mathbf{x}) = inf_{\mathbf{y} \in \alpha} f(\mathbf{x}, \mathbf{y})$도 convex
	- $f(\mathbf{x}, \mathbf{y})$가 y를 고정했을 때 $\mathbf{x}$에 대해서 concave이면 $g(\mathbf{x}) = inf_{\mathbf{y} \in \alpha} f(\mathbf{x}, \mathbf{y})$도 concave

- Lagrangian dual function
	- Lagrangian은 $\mathbf{x}$를 고정했을 때, linear함으로 concave하다(사실 affine이므로 convex, concave 모두 가능함)
	- 따라서 Lagrangian dual function은 concave하다
	- Lagrangian dual problem은 concave function을 maximizing하는 것이므로 사실 convex function을 minimizing하는 것과 같다.
	- **결국 Dual problem은 convex optimization problem이다 $\Rightarrow$ 풀 수 있음이 보장된다.**

### KKT Condition (Karush-Kuhn-Tucker Optimality condition)
$g_i(\mathbf{x}) \le 0, h_j(\mathbf{x}) = 0, \lambda^*_i \ge 0, \lambda^*_ig_i(\mathbf{x})=0, \nabla L(\mathbf{x}^*, \mathbf{\lambda}^*, \mathbf{\nu}^*) = 0$ 일 때 KKT condition을 만족한다고 함.

#### KKT Condition과 관련된 성질
- strong duality가 만족되는 optimization problem의 primal, dual solution은 KKT condition을 만족
- 임의의 optimization problem의 primal, dual solution이 KKT condition을 만족하면 strong duality가 성립한다.