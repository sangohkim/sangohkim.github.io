---
layout: default
title: Matrix Decomposition
parent: Mathematics for ML
grand_parent: LG Aimers
use_math: true
nav_order: 1
---
# Matrix Decomposition
LinearRegression model에서도 SVD를 이용해서 parameter를 구하는 등 ML 모델의 여러 방면에서 사용됨.
## Determinants and Trace
### Determinants Formal Definition

특정 행, 열을 기준으로 Laplace Expansion을 이용하여 정의할 수 있다.

$A \in R^{n \times n}$일 때, 특정 column j에 대해서
$det(\mathbf{A}) = \sum^{n}_{k=1}(-1)^{k+j}a\_{kj}det(\mathbf{A}\_{kj})$

특정 row i에 대해서
$det(\mathbf{A}) = \sum^{n}_{k=1}(-1)^{k+i}a\_{ik}det(\mathbf{A}\_{ik})$

### Related theorems
$det(\mathbf{A}) \ne 0 \iff$ $\mathbf{A}$ is invertible

- pf) $det(\mathbf{A})$와 $det(rref(\mathbf{A}))$는 같이 0이거나 같이 0이 아니다. 그런데 $\mathbf{A}$가 invertible하다면 rref(A)는 identity matrix이니 determinant는 0이 아니다. 

### Related properties
multiple of row/column을 다른 row/column에 더하면 $det(\mathbf{A})$는 그대로 유지됨
- pf) i번째 행에 j번째 행 x $\lambda$인 경우 $det(\mathbf{A}) = \sum^n_{k=1}(a\_{ik} + \lambda a\_{jk})C\_{ik} = \sum^n_{k=1}a\_{ik}C\_{ik} + \lambda a\_{jk}C\_{ik}$ 이므로 $det(\mathbf{A})$와 같다.

특정 row/column에 scalar k를 곱한 경우 $det(\mathbf{A})$는 $k \times det(\mathbf{A})$
- pf) determinant의 laplace extension 정의로 보일 수 있다.

두 개의 row/column을 swap하는 경우 $det(\mathbf{A})$의 부호가 바뀐다
- pf) swap된 row 또는 column을 다시 바꾸기 위해 permutation의 횟수가 1회 증가하므로 부호가 바뀐다.

matrix A, D가 similar한 경우 $det(A) = det(D)$ 
- pf) A, D가 similar한 경우에는 $\mathbf{A} = \mathbf{P}^{-1}\mathbf{D}\mathbf{P}$ 이므로 determinant를 씌우게 되면 같음을 보일 수 있다.

### Trace의 정의
모든 diagonal entry의 합

## Eigenvalue and Eigenvectors
### Properties
$A \in R^{n \times n}$에서 n개의 eigenvalue가 서로 다르다면 A의 eigenvector는 서로 linear independent하며 $R^{n}$ 의 basis를 구성한다.
- pf) n=2인 경우부터 귀납적으로 보인다.

trace는 모든 eigenvalue의 합

determinant는 모든 eigenvalue의 곱

## Cholesky Decomposition
matrix A가 symmetric, positive definite일 때, $\mathbf{A} = \mathbf{L}\mathbf{L}^T$와 같이 A를 decompose할 수 있다.
- L은 lower triangular matrix
- L을 Cholesky factor라고 함
- determinant를 쉽게 계산하는 것 등등에 활용할 수 있음

## Eigendecomposition (EVD)
Coming soon

## Singular Value Decomposition (SVD)
Coming soon