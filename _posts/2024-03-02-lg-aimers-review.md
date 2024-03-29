---
title: '[AI] LG-Aimers 참가 후기'
date: 2024-03-02
permalink: /posts/2024/03/lg-aimers-review/
tags:
  - LG-Aimers
  - AI
  - ML
  - DL
---

# 나의 첫 AI 대회, LG-Aimers 참가 후기

> 2024.03.02. 내가 벌써 3학년...??

  

지난 가을학기가 끝나갈 무렵, 아무 생각없이 2학년을 끝마친 나는 이젠 뭔가 해야한다는 초조함에 시달리고 있었다. 개별연구? 공모전? 어떤것이든 기회가 주어지면 주저없이 해야겠다는 생각으로 여러 대회도 찾아보던 와중에 우연히 광고로 [제 4회 LG-Aimers](https://www.lgaimers.ai/)를 알게되었고 갓생 방학을 보내고자 참가하게 되었다.

### 2023.12

LG-Aimers를 지원할 2023년 12월 당시 나는 AI 쪽으로는 거의 아는게 없던 상태였다. 프로그래밍은 예전부터 꾸준히 해왔지만 인공지능 분야로는 거의 해본적이 없었다. 다만 직전학기에 학부 기계학습 수업을 들어 기본적인 머신러닝, 딥러닝 모델에 관해 개념적으로는 나름대로 탄탄하게 알고 있었다. 내게 부족했던건 개념적인 부분보다는 실제로 라이브러리 등을 사용해서 모델을 구현하는 부분이니, LG-Aimers Phase 2가 시작되기 전에 미리 공부를 해두면 좋겠다는 생각이 들었다.(LG-Aimers Phase 1은 개념 위주라 직전 학기에 들었던 학부 수업이랑 많이 겹쳤다. 그래서 거의 안들었다). 머신러닝에 많이 사용되는 scikit-learn, 딥러닝에 자주 사용되는 Tensorflow, Keras 또는 PyTorch에 대해 잘 설명한 책이 있는지 찾아보다가 Hands-on ML이라는 유명한 책을 발견해 1월 중에 최대한 공부해보기로 했다.

### 2024.01
LG-Aimers는 딥러닝보다는 머신러닝 위주로 운영되는것 같아 HOML에서 머신러닝 부분만 우선 공부해보기로 했다. 다행히 sklearn이 편리하게 사용할 수 있고, 설명 및 문서화가 정말 친절하게 잘 되어 있어서(이때까지 봤던 그 어떤 패키지보다 좋았다) 사용하는게 크게 어렵지 않았다. 직전학기에 학부 수업에서 개념을 열심히 공부했던 것도 큰 도움이 되었는지, 다행히 1월 중으로 Classification, Regression 에 사용되는 여러 모델 훈련 및 전체적인 ML 모델 개발 프로세스를 잘 익힐 수 있었다.

LG-Aimers Phase 2는 개인으로 참가도 가능했지만 팀으로 참가도 가능했다. 우연히 학교에서 LG-Aimers 준비하시는 분들이 있는 오픈채팅을 알게 되었고 그 톡방에 있던 사람들 중 두 분과 함께 셋이서 팀을 이루게 되었다. 팀이 결성된 시기는 1월 마지막 주쯤으로 Phase 2 시작 직전이기에 스터디 같은 건 많이 못하고 데이콘에 있는 이전 LG-Aimers 대회 주제에 대해 수상자 분들의 코드를 분석해보는 스터디 정도만 했다. 이때 머신러닝, 딥러닝을 이용해 문제를 해결해 나가는 과정에 대해 확실히 알게 된 것 같다.
간단히 요약해보자

1. EDA 및 preprocessing
- EDA는 처음에는 용어를 보고 겁이 났지만, 알고보니 그냥 데이터셋 파악하는거였다! 였는데 정의는 간단했으나 실제로 하기에 쉽지는 않았다. 물론 특성 별 분포, 상관관계 등 기본적인 정보를 파악하는건 어렵지 않았지만 너무 단순한것들만 해서 그런지 후술하겠지만 인사이트가 보이지 않았다. Feature engineering은 깊게 들어가니 VIF 등등 생소한 개념, 방법들이 많이 나왔다. 요령도 조금 있어야 하는것 같은데 이런 면에서 아직 데이터분석에는 내가 많이 부족한 것 같다.
2. 모델 설계 및 최적화
- 이 부분은 좀 자신있었다. 분류 문제를 해결한다고 치면 SVM, 여러 앙상블 모델 등등을 적용해보고 성능이 괜찮으면 튜닝으로 최적화하여 좋은 모델을 만들어 내는 과정이다.

### 2024.02
대망의 Phase 2가 시작되었다! 이전까지는 데이콘에서 진행했는데 이번에는 엘리스에서 진행되었다. 주제는 **MQL 데이터 기반 B2B 영업기회 창출 예측 모델 개발**이었다. 우선 주제를 듣자마자 벙쪘던 것 같다. 마케팅 분야에는 문외한이었던 나였기에, 처음에는 주제 및 데이터셋 이해하는 것도 쉽지 않았다. 막막하긴 했지만 아래 과정대로 일단 차근차근 진행해보았다.

#### 1. MQL이 도대체 뭐야
MQL은 Marketting Qualified Lead의 약자로 회사가 투자하면 충분히 상품에 관심을 가질 만한 고객을 의미한다. 즉 이번 모델의 목적은 여러 고객의 정보가 담긴 데이터셋을 받았을 떄 MQL로 전환할 수 있을 만한 고객을 찾는 것이다. 이제 문제가 나름 선명하게 그려지기 시작했다. 이외에도 관련 마케팅 개념을 조금 알아보았으나 글이 너무 길어질 것 같아 우선 넘어가겠다.

#### 2. EDA
일단 csv파일을 DataFrame으로 만들고 아래와 같은 작업을 했다. DataFrame으로만 분석한건 아니고 [Dataprep](https://dataprep.ai/)이라는 자동 EDA툴을 알게되어 기본적인건 Dataprep을 이용했고 세부적인 조정이 필요한 작업만 판다스로 직접 했다.
1. 특성별로 unique한 값 파악하기
2. 결측치, 결측비율 확인하기
3. 수치형 특성이라면 분포 확인하기, 범주형 특성이라면 서로 다른 값들의 비율 확인하기
4. 특성 하나하나 직접 값들을 보면서 겹치거나 불필요한거 찾아내기
5. 특성 사이의 상관관계 분석하기

와 같이 꽤 많은 작업을 했으나 **얻은 인사이트가 거의 없다....** 그나마 알게된걸 정리해보자면
1. 특성 별로 같은 의미의 값이 서로 다르게 기록되어있다
- OTHER, OTHERS
- ETC, ETC.

가 전부였다. 그래서 일단 저것만 적용하기로 하고 모델을 적용해보았다.

#### 3. 모델 적용하기
일단 LogisticRegression을 사용했으나 당연히 택도 없었고 SVM은 너무 느렸고 결과도 별로였다. Imputing 방식도 중간에 KNNImputer, IterativeImputer로 해보았는데 그닥 나아지지 않았다. 베이스라인에서 쓰인 DecisionTreeClassifier를 튜닝해보아도 베이스라인 점수를 넘지 못했다.

팀원분들이 [Autogluon](https://auto.gluon.ai/stable/index.html) 으로 여러 모델을 적용해본결과 앙상블 계열 모델이 가장 성능이 좋았다(이때가 벌써 2월 초중반이었다. 캐글, 데이콘 여러번 해보신 분들은 이때까지 앙상블 안쓰고 뭐했냐 생각하실수도 있는데.....나는 앙상블 계열부터 적용해보아야 했다는걸 이때 처음 알았다ㅠㅠ)

그래서 우선 RandomForestClassifier부터 적용했다. 튜닝도 열심히 해보았지만 베이스라인 못넘었다. ExtraTreesClassifier 등등 다 적용해보았는데 잘 안되었다.

이후 부스팅 계열을 적용해보기로 했다. sklearn에 있던 GradienBoostingClassifier, HistGradientBoostingClassifier는 그리 효과가 좋지 않았다(지금 생각해보니까 내가 제대로 못쓴거 같다). 그 이후 XGBoost, CatBoost, LightGBM을 사용해보기로 했는데 XGBoost를 사용하니까 처음으로 베이스라인을 넘었다!!!! (이때가 벌써 2월 중반)

감격에 겨워 정신을 못차렸지만 우선 진정하고 원인 분석을 해보았다. 보니까 이때까지 사용했던 다른 모델에서 충분한 성과가 나오지 않았던게 물론 모델의 복잡성이 낮은 걸 수도 있겠으나, 내가 class weight 설정을 제대로 하지 않았었던 게 주 원인이기도 하다(주어진 데이터셋은 0, 1이 레이블이었고 비율은 1:11로 엄청 unbalance했다). 그 이후에 XGBoost 미세 튜닝을 최대한 해보았고 public score 0.68까지 도달하였다. 아쉬웠던 부분이 XGBClassifier를 당시에 처음써보아서 파라미터 각각의 역할에 대해 잘 이해하지 못한 상황이라 효율적으로 튜닝하지 못했었다... 나중에 제대로 공부해봐야겠다.

그리고 CatBoostClassifier도 적용해보았는데 이번에는 튜닝도 해보니 0.72까지 퍼블릭 스코어가 올라갔다. 사실 CatBoostClassifier도 범주형 특성에 대한 처리 방식을 잘 이해하지 못해서 그냥 무작정 하이퍼파라미터 튜닝을 한거라 아쉬움이 남는다. 

이후에 전처리 방식도 변경해보고 CatBoost 튜닝도 계속해보았지만 성능향상이 거의 없었다. (여담이지만, CatBoost는 sklearn에 내장된 RandomizedSearchCV로 튜닝하는 것이 쉽지 않았다. fit 메서드에도 인자를 주어야 하는데 이게 지원이 안되는거 같다. 그래서 optuna를 사용하는 것을 추천한다)

마지막 시도로 교차검증하듯이 모델을 튜닝해서 각각의 predict_proba 값을 평균내어 레이블을 결정하는 방식을 시도해서 0.74까지 올랐다. 그리고 Phase 2도 종료되었다.

결과는 첫 시도치고 좋지 않았나 싶다. 우선 private score (final score) 기준 91위로 리더보드에서 밀려나지는 않았다ㅋㅋㅋㅋ public score가 100위였는데 오버피팅을 잘 피한덕분인지 순위가 조금 올랐다.

### 앞으로 해야할 것, 배운점
- XGBoost, CatBoost, LightGBM 개념적으로 이해하기
- Feature engineering 기법 배우기(feature selection, 파생변수 등등)
- unbalanced dataset 처리하는 방법
- sklearn 내장 패키지가 아니거나 모델이 좀 복잡해서 내장 튜닝 패키지 적용이 어려우면 optuna를 쓰자
- 데이콘, 캐글 몇개 더 해보면서 경험 쌓기
