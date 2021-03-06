## 정밀도-재현율 곡선과 ROC 곡선

#### 정밀도
[양성으로 식별된 사례 중 실제로 양성이었던 사례의 비율은 어느 정도 인가요??]에 대한 답을 하고자한다.
	FP이 나오지않는 모델의 정밀도는 1.0
>  암이라고 판정한것중에 얼마나 맞았냐

#### 재현율
[실제 양성 중 정확히 양성이라고 식별된 사례의 비율은 어느 정도 인가요??]

> 전체 암환자중에 얼마나 놓치지 않고 암판정을 내렸느냐 

정밀도와 재현율은 서로 상충하는 관계에 있는 경우가 많습니다
정밀도가 향상되면 대개 재현율이 감소되고 , 반대의 경우도 마찬가지

(https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall?hl=ko)

TP : 스팸을 스팸이라
TN : 정상을 정상이라
FP : 정상을 스팸이라
FN : 스팸을 정상이라


모델의 분류 작업을 결정하는 임계값을 바꾸는 것은 해당 분류기의 정밀도와 재현율의 상충 관계를 조정하는 일이다.


##### 임계값
스팸확률이 0.6인 이메일을 어떻게 처리할까?
이렇게 애매한 값을 이분법으로 확실히 분류를 할 기준이 필요하다. 이 기준을 바로 임계값(Threshold)
로지스틱 회귀값을 이진 카테고리에 매핑하려면 분류 임계값을 정의해야한다. 분류 임계값은 항상 0.5여야 한다고 생각하기 쉽지만 임계값은 문제에 따라 달라지므로 조정해야함

***


90% 재현율과 같은 특정 목적을 충족하는 임계값을 설정하는것은 언제든 가능하지만, 어려운 부분은 이 임계값을 유지하면서 적절한 정밀도를 내는 모델을 만드는일이다.

90% 쟁현율 처럼 분류기의 필요조건을 지정한느것을 **운영포인터(operation point)를 지정한다고 한다.** 라고한다.
운영 포인트를 고정하면 비즈니스 목표를 설정 할때 고객이나 다른 그룹에 성능을 보장하는데 도움이 된다.
**운영 포인트를 고정하면 도움이된다**


#### 정밀도-재현율 곡선
새로운 모델을 만들때 이런 운영 포인트가 명확하지 않은 경우가 많다. 
그래서 문제를 더 잘 이해하기 위해 모든 임계값을 조사해보거나 , 한번에 정밀도나 재현율의 모든 장단점을 살펴본느것이 좋은데 
이를 위해 정밀도-재현율 곡선을 사용한다.
(가능한 모든 임계값(결정함수에 나타난 모든 값)에 대해 정밀도와 재현율의 값을 정렬된 리스트로 반환한다)
곡선의 각 포인트는 decision_function의 가능한 모든 임계값



==P.351 precision_recall_curve 함수는 가능한 모든 임계값(결정 함수에 나타난 모든값) 에 대해 정밀도와 재현율의 값을 정렬된 리스트로 반환한다==

jupyter



## ROC와 AUC

Accuracy의 단점을 보기위해 Precision과 Recall을 사용했는데 ROC도 마찬가지로 좀더 자세히 보기위해서 사용

ROC곡선은 여러 임계값에서 분류기의 특성을 분석하는데 널리 사용하는 도구.
정밀도-재현율 곡선과 비슷하게 ROC곡선은 분류기의 모든 임계값을 고려하지만 , 정밀도와 재현율 대신 
**진짜 양성비율(TPR)에 대한 거짓양성비율(FPR)**을 나타낸다.
> 진짜 양성비율은 재현율의 다른이름이며 , 거짓양성비율은 전체 음성 샘플 중에서 거짓양성으로 잘못 분류한 비율이다.

TPR = TP / TP+FN
FPR = FP / FP+TN

TNR = TN / TN +FP

binary classification 과 medical application에서 많이 쓴다

**AUC** 는 정밀도-재현율 곡선의 아랫부분 면적 계산할때 슨 평균 정밀도와 같은 , 곡선 밑에 면적계산하는것

AUC가 넓으면 이 피쳐들이 좋은 피쳐다 이 판단들이 믿을만한,안정적인 판단이다라는것을 알수있는것

Jupyter


![](https://github.com/wnsghek31/machine-learning-/blob/master/사진.PNG)

어떤부분은 확실히 구분되는 부분이있지만
어쩔수없이 불분명한 부분이 있을수바께없는데 그부분에서 최선의 판단을 내려한다. (초록색라인같은)
[ 어쩔수없이 에러들을 포함을한다. ]

밑에가  더 분리하기 쉬운데 
이차이를 구분을 하는게 ROC 커브이다. 
어떤 판정을 내리는데 이 판정선이 조금 움직이면 어떤 안좋은 결과를 초래하느냐에 대한 얘기가 ROC커브
  
밑에같이 초록색 선이 약간 바뀌더라고  mis classification이 많이 안생기면 빨간 크래프를 그리는것
만약에 (두분포가) 많이 겹친다면 직선이 나오는것

좋은 feature인지 아닌지를 구분할때 사용

ex)
누가더 잘달리냐의 예측을 하는데
키와 몸무게를 피쳐로 골랐는데 ,  키큰사람중에 느린사람 키큰사람중에 빠른사람들이 겹쳐서 많은 부분들이 겹칠거야
좋은피쳐인 지난대회 성적을 쓰면 , 100등안에는 잘뛰는사람 100등 밑은 못뛰는사람으로 고르면 밑에같은 그래프가 나옴 (110등 90등으로 잘라도 그렇게까지 sensitive 하지않다)

medical paper나 machinlearning paper에서 ROC커브를 많이쓴다. 커브가 빨간색처럼생겼다면  어떤 decision boundary에 덜 민감하고 잘 분리가 되있다. 이 A와 B를 구분하는게 믿을만한다.
(이 decision boundary ~~ 하면 ROC커브가 저리나오고 저리나오면 AUC가 크니까 좋은 피쳐다.)

==ROC 곡선은 다양한 분류 임계값의 TPR 및 FPR을 나타냅니다. 분류 임계값을 낮추면 더 많은 항목이 양성으로 분류되므로 거짓양성과 참양성이 모두 증가==


이 초록색을 오른족으로가면  TP가 줄어들고 Negative은 다 검출할수있을것
왼쪽으로 가면	positive는 다 검출할수있다.

TPR = TP / TP+FN
FPR = FP / FP+TN
TNR = TN / TN +FP

TPR 은 전체 postive중에 얼마나 빠짐없이 검출해냈느냐
TNR 은 전체 negative중에 얼마나 빠짐없이 검출햇냐
선이이동할수록 둘이 trade-off가 생김

그래프 처음이 (b) , 그래프 맨앞이 (a)



## 다중 분류의 평가 지표

다중 분류를 평가하는 지표는 기본적으로 이진 분류 평가 지표에서 유도되었다.
다만 모든 클래스에 대해 평균을 낸것. 다중 분류의 정확도도 정확히 분류된 샘플의 비율로 정의한다.

그래서 클래스가 불균형 할때는 정확도는 높은 평가 방법이 되지 못한다.

## 회귀의 평가지표

일반적으로 R^2 이 회귀 모델을 평가하는데는 좋은 지표이다.
가끔 평균 제곱 에러나 평균 절댓값 에러를 사용하여 모델을 튜닝할때 이런 지표를 기반으로 비지니스 결정을 할수있다.
그러나 일번적으로 R^2

## 모델 선택에서 평가 지표 사용학;ㅣ

GridSearchCV나 cross_val_score를 사용하여 모델을 선택할대 , AUc같은 평가지표를 사용하고 싶은 경우가 많이있다.
사용하려는 평가지표를 문자열로 넘겨주기만 하면됨



빨간색처럼 생겻다하면 decision boundary에 덜 민감하고 잘 분리가 되엇다 라고 판단내릴수있늑넛.


암진단에서 그냥 모두 정상이라 판정하면 정확도 높게나와




