---
title: '[AI] Imputation Process for AI'
date: 2024-03-11
permalink: /posts/2024/03/imputation-process/
tags:
  - AI
  - Imputation
---

AI 대회에서 활용하기 위해 여러 imputation 기법들을 정리해보았습니다.

> 2024.03.10
LG-Aimers 4기에 참가하면서 결측치 처리에 대한 저만의 명확한 프로세스가 없다는 것을 깨달았습니다. 물론 다양한 데이터셋에 따라 적절히 여러 방식을 적용하는 것이 필요하지만 어느정도 체계를 잡고 싶은 생각에 결측치 처리 방법 및 프로세스를 정리하였습니다.
> 

# 결측값의 유형

> 데이터가 결측될 확률에 따라 아래와 같이 3가지로 분류합니다.
현실적으로 실제 상황에서 결측치를 아래 3가지 중 하나로 명확히 분류하는 것은 어려운 것 같습니다.
> 

## MCAR (Missing Completely At Random)

데이터가 결측될 확률이 모든 경우에서 같을 때 MCAR로 분류합니다. 즉 다른 feature가 어떤 값을 가지든, 혹은 결측된 데이터의 값이 어떻든 결측될 확률에 영향을 주지 않는 경우를 의미합니다.

- 데이터를 기입하는 과정에서 실수로 누락시켰거나 전산 오류가 발생하여 기입되지 않은 경우가 MCAR에 해당합니다.
- 결측 원인이 데이터셋의 다른 값들과 전혀 영향이 없기 때문에 다른 feature, 같은 feature 내의 관측된 값으로 imputation을 진행하거나 결측된 데이터를 제거해주시면 됩니다.
- 가장 이상적이며 편리한 경우입니다.

## MAR (Missing At Random)

데이터가 결측될 확률이 같은 feature 내의 observed data 내에서 동일한 경우에 MAR로 분류합니다. 즉 다른 feature의 값에 따라 데이터가 결측될 확률이 높아지거나, 낮아질 수 있지만 결측된 값과는 관련이 없을 때를 의미합니다.

- 몸무게를 측정할 때 젊은 여성 군집에서 몸무게 결측값이 많이 나온 경우 MAR로 분류할 수 있습니다.

## MNAR (Missing Not At Random)

MCAR, MAR이 아닌 경우에 MNAR 또는 NMAR이라고 합니다. 즉 데이터가 결측될 확률이 다른 feature의 값, 결측된 그 자신의 값 또는 데이터셋에 기록되지 않은 외부 요인의 영향을 받을 경우를 의미합니다.

- 흡연 여부를 선택할 때 실제 흡연자가 사회적 인식을 고려해 사실대로 응답하지 않고 응답란을 비워두는 경우가 예시가 될 수 있습니다

## MAR, MCAR, MNAR 인지 판단하기

결측값이 MAR, MCAR, MNAR인지 구분해주는 명확한 방법은 없습니다. 우선 모든 결측값을 MNAR이라고 기본적으로 가정한 뒤, 데이터셋에 대해서 알고 있는 사전 정보, 데이터셋 수집 방법을 통해 MAR, MCAR을 구분해 나가야 합니다. 

데이터의 결측여부를 레이블로 하는 Logistic Regression 모델을 fitting해보는 방식으로 insight를 얻을 수도 있습니다. 결측 여부와 관련이 있는 feature를 찾게 된다면 MAR일 가능성이 높고 그렇지 않다면 MCAR, MNAR 중에서 구분하는 방식으로 판단해볼 수 있습니다.

## MAR, MCAR, MNAR에 따른 처리 방법

### MCAR인 경우

Complete case analysis (결측값이 없는 샘플만 남겨놓기) 또는 Simple Imputation, Multiple Imputation 등등 적절히 imputation 방법을 적용하면 됩니다.

### MAR인 경우

MCAR만틈 valid하지 않지만 complete case analysis도 적용 가능합니다. Simple Imputation도 적용가능하며 Multiple Imputation또한 valid 합니다.

### MNAR인 경우

결측값이 발생한 외부적인 요인이 없는지 여부를 조사하고 결측값에 대한 모델링이 필요합니다. 그리고 모델링 후 Multiple Imputation을 적용해볼 수 있습니다. 그러나 MNAR 결측값은 사실상 데이터셋 내부의 정보로 제대로 처리할 수 없기 때문에 일반적으로 MNAR 결측값인 경우 MAR로 가정하고 처리하는 것이 좋을 것 같습니다.

# 결측값 처리 방법

<aside>
💡 MCAR, MAR로 가정하였습니다.

</aside>

## 결측값의 비율 확인하기 (공식적인 가이드라인이 아닙니다!)

| 결측 비율 | 처리 방법 |
| --- | --- |
| ~10% | 제거하기 또는 여러 방법의 imputation 적용하기 |
| 10% ~ 20% | Hot deck (KNN Imputation 등등), Regression, Model-based method imputation |
| 20% ~ 50% | Model-based method, Regression imputation |
| 50% ~ | 해당 feature 제거하기 |

## 0. 결측값 그대로 두기

결측 비율이 그리 크지 않고 XGBoost, LightGBM과 같이 결측치를 자동으로 처리하는 기능을 제공하는 모델을 사용하는 경우 그대로 두는 것도 하나의 방법이 될 수 있습니다.

## 1. 결측값 제거하기

데이터셋에 따라 편향, 분산을 고려하여 적절한 방법을 사용해야 합니다.

- 결측치가 너무 많은 feature 제거하기
- 결측치가 있는 row 제거하기

## 2. Imputation

### 2-1. Simple Imputation

단일 측정값을 이용해서 결측값을 대체하는 방법입니다. 구현이 간단하고 Multiple Imputation에 비해 처리 시간이 빠르고 자원을 많이 요구하지 않는다는 장점이 있습니다. Imputation을 할 때 가장 기본적으로 적용시켜보는 방법이며 더 복잡한 Imputation이 필요할 때 Multiple Imputation을 사용합니다.

- mean, median, most frequent imputation
- constant imputation
    - 0, -1 로 채우기 등등
- KNN imputation
    - sklearn KNNImputer 이용하기
    - outlier에 민감하므로 이상치를 미리 확인해보는 작업이 필요할 것 같음

### 2-2. Multiple Imputation

<aside>
💡 MICE 외에도 MVNI 기법이 있지만 우선은 MICE에 대해서 알아보고 나머지는 추후에 알아보도록 하겠습니다.

</aside>

Simple Imputation을 여러번 반복하여 결측값을 채우는 방법. 일종의 앙상블이라고 생각하면 됩니다. 결측비율이 꽤 높거나, MAR, MNAR에 가깝다고 생각되는 feature에 대해 적절히 적용하면 좋을 것 같습니다.

![[Stef van Buuren, Karin Groothuis-Oudshoorn (2011). “mice: Multivariate Imputation by Chained Equations in R”. Journal of Statistical Software](https://www.jstatsoft.org/article/view/v045i03)](/docs/AI/Untitled.png)

[Stef van Buuren, Karin Groothuis-Oudshoorn (2011). “mice: Multivariate Imputation by Chained Equations in R”. Journal of Statistical Software](https://www.jstatsoft.org/article/view/v045i03)

- MICE의 절차
    1. 데이터셋 생성(imputed data): 결측치를 특정 strategy를 이용해서 모두 대체한 데이터셋을 m개 생성합니다.
    2. 분석과 추정: 임의로 결측값을 대체한 데이터셋을 estimator로 분석합니다.
    3. 추정치 합치기
- IterativeImputer
    - scikit-learn에서 MICE에 영감을 받아 구현된 imputer
    - MICE 방식을 거의 동일하게 적용할 수 있음

### 2-3. Deep Learning (DataWig) 이용하기

AWSLAB에서 [Biessmann, Salinas et al. 2018](https://dl.acm.org/citation.cfm?id=3272005)를 이용하여 구축한 imputation 패키지입니다.. 범주형 특성에도 적용이 가능하며 다른 방식에 비해 비교적 정확하다고 합니다.. 추후 캐글에서 사용하며 구체적인 방식을 알아보겠습니다.