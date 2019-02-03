
variance가 큰것은 샘플들에 많이 의존된것 그래서 1 샘플에서의 그려진 그래프와 , 2 샘플에서 그려진 그래프의 차이가 크게나타남.

bias는 비슷한 그래프가 그려지지만(variance 적다) 항상 비슷하게 그려지니까 피할수없는 에러들이 있게됨. 그러한 틀린값들이 많은게 bias가 큰것.


텐서플로우하다


Variable W1 already exists, disallowed. Did you mean to set reuse=True or reuse=tf.AUTO_REUSE in VarScope? Originally defined at
에러는

tf.reset_default_graph()

# 1

## Convolution layer

![](https://github.com/wnsghek31/machine-learning-/edit/master/deeplearning.ai/1.PNG)

첫번째로 하는일이
이미지를 보고 vertical edges , horizontal edges 를 찾는것 


convolution이 이 사진과 filter의 각 해당하는 행렬 부붙 곱해서 그값들 다 더하는것

1,0,-1
1,0,-1
1,0,-1   필터를 적용해서 vertical edge detection을한다.

이런 convolution 하는걸  tf.nn.conv2d

![](https://github.com/wnsghek31/machine-learning-/edit/master/deeplearning.ai/2.PNG)


결과의 흰색부분이 원본사진의 중간선을 의미하는것이다. (이미지 가운데 강한 경계선이 있다는것을 알려주는) 
조금 두껍긴하지만 이것은 굉장히 작은 이미지를 예로 하고있기때문에 그런것
몇천*몇천의 이미지를 쓰면 꽤나 정교하게 수직 경계선을 찾아낸다.

3(10*1)와 3(0*-1) 로인해 값의 차이가 커져서 선이 있는걸 아는것 

1,1,1
0,0,0
-1,-1,-1 은 가로 필터


1,0,-1
2,0,-2
1,0,-1 은 sobel filter라고하며.. 여러가지 필터들을 만들어놨따.

그런데 복잡한 이미지에서 윤곽선을을 검출하려고 할때 , 
9개의 숫자를 일일이 고를 필요가 없다는것을 알게됨. **이숫자를 학습시킨다!**
그랬더니 가로 세로 윤곽선 말고 , 43도 70도 기울어진 윤곽선 같은것도 검출할수있게됨
이 변수를 학습함으로써 신경망이 윤곽선 같은 하위 단계의 속성을 학습할수있게됨.
 
### 패딩 

nxn 이미지 , fxf 필터  convolution
결과 : n - f + 1 * n - f + 1

#### convolution만 했을때의 단점

* 이미지 크기가 작아져서 계속 계산하다보면 너무작아짐 1x1로..

* 가장자리 픽셀을 보면 결과아미지에 단한번만 사용하기에 (중간에있는것들	은 여러개의 3x3영역에 걸쳐있음) 그만큼 결과 이미지에 덜 사용하는	것이기에  **가장자리 근처의 정보들은 거의 날려버리게되는것**.
  
이를 보완하기 위해 패딩을 더해주는것이다
 
결과 : n +2p - f + 1 * n + 2p - f + 1

p = (f-1) /2 로하면 크기를 보존할수있다.
**일반적으로 거의 항상 f는 홀수이다.** 
컵퓨터 비전에서는 짝수의 필터를 거의 볼수 없음.

* f가 짝수이면 패딩이 비대칭이됨.
* 홀수 필터는 중심 위차가 존재함

컴퓨터비전의 관습에는 3x3 필터가 매우 일반적이고 , 5x5 , 7x7 등의 필터를 사용
(짝수의 f를 사용해도 괜찮은 성능이 나오긴한다)

### 스트라이드

stride 까지 넣으면

( n + 2p - f )/s + 1 * ( n + 2p - f )/s + 1
만약 이 값이 정수가 아니라면 내림을 해준다.

### 필터

RGB이미지 
6x6x3 ( 3개의 채널 )

3x3x3의 필터를 가져야함.  **이미지의 채널수와 필터의 채널수는 같아야하니까**

3x3x3 필터니까 27개 숫자를 rgb채널에 각각 곱해주고 (채널당 9) 더해서 값을 얻는다.

빨간색의 세로 윤곽선을 찾으려고한다면
빨간필터  파란필터    초록필터
10-1     000        000
10-1     000        000
10-1     000        000
이렇게 합쳐서 3x3x3 필터를 이루게 되는것.

모든색의 윤곽석을 검출하는 세로 윤곽선 검출기
빨간필터  파란필터    초록필터
10-1     10-1     10-1
10-1     10-1     10-1
10-1     10-1     10-1

**6x6x3  *  3x3x3  하면  4x4x1 이 나온다.
이런 3x3x3 을 두개쓰면 4x4x2 가 되는것**
 
가로와 세로 윤곽선처럼 두개의 특성 또는 수백개의 특성들을 검출할수있고 그 결과는 **검출하고자 하는 특성의 수만큼 채널을 가지게 되는것** 
(여기서 체널은 결과가 4x4x1 나온거의 개수를 의미한다 RGB채널개수가아님 위에서 4x4x2에서의 2값)

![](https://github.com/wnsghek31/machine-learning-/edit/master/deeplearning.ai/4.PNG)

6x6x3 --> 4x4x2 (2개의 filter니까(3x3x3))

채널은 마지막 크기를 나타내고 3D 입체형의 깊이라고도 불림. 즉 추출하고싶은 특성(?) 의 수이기도하다.( 가로 , 세로 , 60 , 30도 윤곽선 등..)

27 parameter + 1 bias가 10개 있으니까 280개의 변수를 갖는다.
입력이미지의 크기가 1000x1000 이거나 5000x5000이거나에 상관없이 280개의 변수이다.  열개의 채널인거지!!(필터의 개수니까)
아주 큰 이미지라도 적은 수의 변수로 가능 **이것이 과대적합을 방지하는 CNN의 한 특징**

![](https://github.com/wnsghek31/machine-learning-/edit/master/deeplearning.ai/5.PNG)

convolution layer 서의 
convoution ,bias , ReLu ,하는게 fully connected layer에서의 y=ax+b에다가 ReLU 취해주는거랑 같다.

f^[l]로 필터 크기를 나타냄 (fxf)
[l]은 특정 계층l을 나타내는 표현

채널의수 nc는 필터의 수이다. 

![](https://github.com/wnsghek31/machine-learning-/edit/master/deeplearning.ai/6.PNG)

Convolution layer의 작업들 

7x7x40 이 남는데 이걸 펼쳐서 1960개의 노드로 만들어서 (즉 하나의 벡터로 만드는것) logistic 이나 softmax 넣으면 결과값이 나온다.

신경망이 깊어질수록 width , height는 줄어들고 , channel 수는 늘어난다.


## Pooling layer

최대풀링은 4x4 입력이 어던 특성의 집합이라면 (아닐수도 있지만)
이 4x4를 어떤 특성이나 어떤 신경망 층의 활성값이라고 할때 가장 큰 수가 특정 특성을 의미 할수있따. 분명히 어떤 부분에는 어떤 특성이 존재할수도 있고 , 없을 수도있다. 그래서 최대 연산은 이러한 특성이 한곳에서 발견되면 그것을 결과로 내는것.
그래서 한 특성이 필터의 한부분에서 검출되면 높은 수를 남기고 특성이 없다면 그 사분면 안의 최대값은 여전히 다른것들에 비해 작은 수일것.(다른 데 특성이있을때의 최대값 보단 작을테니)

**max pooling은 실험속에서 이게 성능이 좋다** , 정확한 이유는 알지못해..
max pooling 에는 학습 할수 있는 변수가 없다. gradient descent로 학습할수가없어. f와 s (filter의 크기 , 스트라이드) 가 고정이라서 학습이안되.
fix function , nothing to learn

**정하는 파라미터는 f , s 로  f=2 , s=2 가 자주 사용된다.**
max poolin에서는 패딩 거의 안쓴다. 
max pooling이 average pooling 보다 훨씬 많이 사용되지만 ,  하나의 예외는 신경망 아주 깊은 곳에서 평균 풀링을 사용해서 7x7x1000의 값을 1x1x1000 으로 만든다.


![](https://github.com/wnsghek31/machine-learning-/edit/master/deeplearning.ai/7.PNG)

5x5x16 -> 400개의 노드 로 만들고 이걸 120개의 노드로 만들면 이 120개있는 층이 첫번째 fully connected layer 이다. 
왜나햐면 400노드와 120노드가 모두 연결 되있기 때문에. (단일 신경망 층과 유사하다.)
84개 노드를 softmax를 취해서 출력이 10개가 되면 mnist 숫자 인식기




### CNN의 하이퍼파라미터 정하기

신경망을 디자인하는 대부분의 일은 필터의 크기 , 스트라이드나 패딩 필터의 개수 같은 하이퍼 파라미터를 선택하는 과정이다.

하이퍼 파라미터 결정 지침
* 하이퍼 파라미터를 직접 선정하지 말고, 문헌에서 다른 사용자들에게 작동했던 하이퍼 파라미터를 보고 내꺼에도 잘 작동할 구조를 선택하자.





## fully connected layer 사용보다 conv layer 사용의 이점

![](https://github.com/wnsghek31/machine-learning-/edit/master/deeplearning.ai/9.PNG)

만약 위에껄 fully connected layer로 연결한다면 3072* 4704개의 변수.
그러나 Conv layer는 필터에는 (5*5 + 1) 개의 변수를 갖고 6개 있으니 156개의 변수를갖는다.

### Parameter Sharing , Sparsity of Connection

![](https://github.com/wnsghek31/machine-learning-/edit/master/deeplearning.ai/8.PNG)

이렇게 적은 변수 갖는 이유가 하나는 parameter sharing
필터들이(세로 윤곽선 같은 ) 이미지에 각부분에 움직이면서 연산하기에 이미지에 여러부분에서 쓰임. 그렇기에 적은 파라미터가 드는거 .
>ex. 세로 검출기는 왼쪽위와 오른쪽위에 같이 똑같이 적용해도 둘다 세로를 뽑아내기 위함이므로 공유해도 되. )
이게 parameter sharing


두번째는 희소 연결 때문이다.
왼쪽 초록색 9칸이 오른쪽 초록색 1칸을 만든다. 
나머지 픽셀값들은 이결과값에 아무런 영향을 주지않아. 다른 부분도 마찬가지

이 두가지 방법으로 신경망의 변수가 줄어서 작은 훈련 세트를 가지게함. 과대적합 또한 방지 

==transaltion invariance 감지에도 좋다. ? 잘몰겟음==



# 2주차


**하나의 컴퓨터 비전 작업에서 잘 작동하는 신경망 구조는 다른 작업에서도 잘 작동하는 경우가많다 **
만약 누군가 발명해낸 신경망 구조가 고양이 , 개나 사람을 인식하는데 좋다면 , 자율 주행차 개발등의 다른 비전 작업에도 이 신경망 구조를 적용할수있따.

LeNet-5 , AlexNet , VGG , ResNEt , Inception
논문들의 개념은 도움이 된다. 어떻게 효과적으로  신경망을 구축할지 감을 잡을수있따.


그림들을 종이에다 그려놨따.


## VGG-16
vgg-16 의 구조적 장점은 꽤나 균일하다는것 
몇개의 conv layer 뒤에 pooling layer 해서 높이 너비 줄여주고, 필터의 개수가 2배수로 늘어난다. (아마 충분히 크기에 더이상 늘리지않은듯)

이 상대적 획일성의 단점은 훈련시킬 변수의 개수가 너무 많다는것 

채널수는 두배로 증가하고 , 너비와 높이는 1/2로 된다. 상당히 체계적





## Vanishing , Exploding Gradient

skip connection
한층의 활성값을 가지고 훨씬 깊은 층에 적용하는 방식
ResNet에서쓰임

![](https://github.com/wnsghek31/machine-learning-/edit/master/deeplearning.ai/10.PNG)

그림이 적철치 않다 a[l]이 없어서.

원래는 main path(그냥 검은 화살표)를 거쳐서 가야하는데 , 신경망의 더 먼곳 까지 단번에 가게 만드는것 (ReLU 비선형성을 적용해주기 전에 a^[l]을 더해줌) 이 길을 short cut(파란색선)이라고 함.
a^[l]이 short cut으로 인해 정보가 신경망의 더 깊은곳으로 갈수있다.
그림 밑에 처럼 a^[l]이 a^[l+2] 연산을 위한 ReLU에 더해지고 이 a^[l]을 residual block 이라고 한다.

Residual block을 사용하면 신경망을 훨씬 더 깊게 훈련 시킬수잇다.

![](https://github.com/wnsghek31/machine-learning-/edit/master/deeplearning.ai/11.PNG)

==그림에서 처럼 w[l+2]a[l+1] + b[l+2] 가 0이 되면 추가한 a[l]은 0이상의 값이기 떄문에 손해가 없는거같음. 0이나 작은값으로 가면 안좋은데 선형 함수에서 0이 나와도 뒤에 붙혀둔 a[l]이 그것을 0이 아니게 할 가능성이 크니까 성능이 좋아지는게 아닐까?? 자세한 기술적 내용은 잘 모르겠따.==

추가된 층이 항등 함수를 학습하기 용이하기 떄문이다. 성능의 저하가 없다는것을 보장 할수 있꼬 운이 좋다면 성능을 향상 시킬수도 있는것
기본적으로 성능의 저하가없고 , 경사하강법으로 더 좋아지기만하는것

여기서 z[l+2] + a[l]은 두개가 같은 차원을 가진다고 가정하는것이다 그래서 ResNet이 same convolution을 많이한다.
==차원이 같다는것은 높이 너비가 같다는거 맞겟지 채널은 안같아도대?? ==
만약에 차원이 다르다면 a[l]에 Ws 라는 행렬을 곱해줘서 차원이 같아지게 해준다.



## 1x1 합성곱

네트워크에 비선형성을 더해준다.
채널의 수를 조정할수있게 해준다.

6x6x1 처럼 1채널이아닌 6x6x32 같이 여러 채널이 있을때 1x1 필터와 합성곱을 하는것을 훨씬 의미가 크다.

![](https://github.com/wnsghek31/machine-learning-/edit/master/deeplearning.ai/1x1.PNG)

6x6x32 에 1x1x32 필터를 곱한다면 
6x6에서의 한칸 (32채널이니까 그 같은위치에 값은 32개있는것) 그러니 32개의 숫자들이 각각 곱해져서 더해진다. 그뒤에 ReLU 취하면 한점이되서 나온다 (6x6 에서의 한점)

필터가 여러개면 그런 6x6이 여러개 나오는거고
fully connected neural network를 36개의 위치에 각각 적용해서 32개의 숫자를 입력값으로 받고 필터의 수만큼 출력 하는것.

==1x1xn 필터를 하나씩 쓸때마다 같은 width, height의 채널이 한개씩 생기는것. 원본데이터의 같은위치인데 여러채널에있는 값을 하나로 압축할수있는 그런 느낌??==


28x28x192 이 있따면 width 와 height를 줄이고 싶다면 풀링을 사용한다. 
그러나 채널을 줄이고싶을떄는 ?? covnlayer 1x1x192의 필터를 32개 해주면 28x28x32가 나온다.

물론 채널을 안줄이고 1x1x192 필터를 192개 해서 채널 유지해도 된다
**1x1 합성곱의 효과는 비선형성을 더해주고 하나의 층을 더해줌으로써 더 복잡한 함수를 학습할수있따. 채널을 줄여줄수도있꼬**



## 인셉션 네트워크 (GoogLeNet)
네트워크가 복잡해 성능이 뛰어나다
1x1 , 3x3 ,5x5 또는 풀링층인지 고민하기 원치 않아서, 그것들을 전부 다 실행해서 함께 엮는것이다.
계산비용은 1x1 필터를 통해 줄인다.

즉 필터의 크기나 풀링을 결정하는 대신 그것들을 전부다 적용해서 출력들을 다 엮어낸뒤 , 네트워크가 스스로 원하는 변수나 필터 크기의 조합을 학습하게 하는것

![](https://github.com/wnsghek31/machine-learning-/edit/master/deeplearning.ai/12.PNG)

다른 크기의 필더들은 same convolution 해줘서 28x28이되게한다.
맥스풀링또한 28x28 이 되게 해줘야한다. 

입력은 28x28x192고 출력은 32+32+128+64 로 28x28x256의 출력이다.

문제는 계산비용이여

![](https://github.com/wnsghek31/machine-learning-/edit/master/deeplearning.ai/13.PNG)

위에 5x5가 나오게 연산하려면

5x5x192 필터 32개에다가 아웃풋인 28x28x32

5*5*192*28*28*32 로 1억2천만개나 되는 변수가나온다.
(필터 변수개수 * 아웃풋 변수개수)

![](https://github.com/wnsghek31/machine-learning-/edit/master/deeplearning.ai/14.PNG)

그런데 1x1을 쓰면

28x28x192에다가 1x1x192 필터 16개 해줘서 28x28x16 나오면
여기다가 5x5x16 32개 해주면 28x28x32 가 나온다..

왼쪽의 192나 되는 큰 볼륨을 16으로 줄이고 ( 이층을 bottleneck layer 라고함) 계산.

1x1x192 필터 16개를 사용하기에 28x28x16 출력을 계산하기 위한 비용은   
28*28*16*192   = 240만
(아웃풋변수 개수 * 필터변수 개수)

두번째 합성곱층의 비용은  28*28*32 * 5*5*16 = 1000만
그냥 하는거보다 1/10 바께 안든다.

**shrinking down the representation size 하는것이 성능에 지장에 줄거라는 걱정 할수있는데 , bottle neck layer를 적절히 구현한다면 representation size 크기를 줄이는 동시에 성능에 큰 지장없이 많은 수의 계싼을 줄일수있따.**


![](https://github.com/wnsghek31/machine-learning-/edit/master/deeplearning.ai/15.PNG)

==activation or output from some previous layer를 인셉션 모듈의 인풋으로 받는다는데 이것들이 정확히 모지??==


이것이 하나의 인셉션 모듈이다. 따로따로 conv pool 하고나서 엮어내는것.(같은크기의 여러개 채널들이 붙는..)
인셉션 네트워크는 이런 모듈을 모아놓은것이야.

pooling layer 하면 채널수는 그대로이니까 1x1 conv 해서 채널수도 줄여주는것.

![](https://github.com/wnsghek31/machine-learning-/edit/master/deeplearning.ai/16.PNG)

중간중간 POOL이 적용되있는 인셉션 모듈이 있낀하지만 이런 기본적인 인셉션 모듈이 네트워크 상에서 반복되는 이런게 인셉션 네트워크
밑에 곁가지들은 은닉층을 가지고 예측을 하는것
자세히보면 fully connected layer 지나서 softmax 취하는 맨마지막 작업을 저기서도 해주는거야
네트워크게 정규화 효과를 주고 과대적합을 방지해준다.
==마지막 결과와 은닉층 결과를 비교해서 과대적합을 방지할수있을것같은데, 정규화 효과는 어덯게 주는거지??==

인셉션 모듈의 업그레이드 된 버전은 ResNet에 스킵연결을 활용하여 훨씬 더 좋은 성능을 보여주기도한다
**이렇게 다 섞어쓰면 좋은게 나오는..**







