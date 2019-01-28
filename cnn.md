## CNN

CNN에서는 FC 대신 Convolutional layer 와 pooling layer를 activation function 앞뒤에 배치한다
출력에 가까운 층에서는 FC Layer를 사용할수있다 또한 마지막 출력 계층에서는 FC - SOFTMAX 그대로 간다

**전형적으로는**
```
[(CONV - RELU)*N (- POOL) ]*M -(FC - RELU)*K - SOFTMAX 
  N is up to ~5 , M is large , 0 <= K <= 2
```  
Pooling layer는 생략하기도한다. CONV - RELU - CONV - REUL - POOl 같이

## Fully Connected Layer
이전 계층의 모든 뉴런과 결합되어 있었고, 이를 Affine layer라고 불렀다. 이렇게 이전 계층의 모든 뉴런과 결합된 형태의 layer를 FC layer라고 한다.

(좀찾아보자)


Fully Connected Layer (Dense Layer) 만으로 구성된 인공 신경망의 입력 데이터는 1차원(배열) 형태로 한정됩니다. **한 장의 컬러 사진은 3차원 데이터** (rgb를 예로들면 r g b 각각 256x256 배열이 세개있으니까)입니다. 배치 모드에 사용되는 여러장의 사진은 4차원 데이터입

사진 데이터로 CNN안쓰고 Fully Connected 신경망을 학습시켜야 할 경우에, 3차원 사진 데이터를 1차원으로 평면화시켜야 합니다. 사진 데이터를 평면화 시키는 과정에서 공간 정보가 손실될 수밖에 없습니다  
이미지의 공간 정보를 유지한 상태로 학습이 가능한 모델이 바로 CNN(Convolutional Neural Network)

> MNIST 이미지는 (1채널,가로28픽셀,세로28픽셀) 인 3차원 데이터인데 FC Connected Layer만으로 이루어져있으면 (1,784)의 1차원 데이터로 평탄화 해서 넘겨야한다.
(1,784)는 2차원 배열이지만 1은 이미지 1개를 의미하므로 , 이미지 1개에 데이터는 784에 들어있는것이므로 이미지 자체는 1차원 데이터 784


### Convolution Layer
입력 데이터에 필터를 적용 후 활성화 함수를 반영하는 필수 요소
(행렬곱 WX+b  필터당 bias 1개   WX하면 scalar 값나올테니까?   bias도 (,,1)로 scalar값)  

이미지에 필터를 이동시키면서 이미지와 곱한 결과를 적분해나간다
* ==여기서 수행하는 적분(덧셈)은 단일 곱셈-누산(fused multiply-add)라한다==
* bias 는 필터를 적용한 결과 데이터에 더해진다. 항상 1x1이며 이 값을 모든 원소에 더한다

필터로 연산을 할때 출력에 영향을 미치는 영역이 지역적으로 제한되어 있기까까때문에 이는 지역적인 특징을 잘 뽑아낸다는것이고 영상 인식에 적합한것

> 필터로 연산할때 전체 이미지중 부분적인 곳만 보고 여러번 연산하니까

예를 들어 코를볼때 코 주변만보고 , 눈을볼때 눈 주변만 보면서 학습 및 인식하는것
.

##### 패딩  
합성곱 연산을 거칠때마다 크기가 작아지게 되는데 출력 크기가 너무 줄어드는것을 막기위해 !
데이터의 사이즈를 보전시키고 , 데이터의 사직 끝을 표시할수있따.


필터개수는 32 , 64 ,128 , 512개 같이 2의 배수로 슨다  512개 젤 많이쓴다
필터 들의 w들은 xavier initializer 나 랜덤값으로 초기화 시키고 . 사진을 통과시키고 backpropagation을 하는것

거의 POOLING 하고나면 사이즈를 절반으로 줄이는 형태를 가지고 한다.

##### Flatten Layer
이미지의 특징을 추출하는 부분과 이미지를 분류하는 부분 사이에 이미지 형태의 데이터를 배열 형태로 만든다
CNN에서 CONV Layer나 Pooling Layer를 반복적으로 거치면 주요 특징만 추출되고, 추출된 주요 특징은 전결합층(FC Layer) 에 전달되어 학습된다.
**CONV와 POOl 레이어 에서는 주로 2차원 자료를 다루지만 FC에 전달되기 위해서는 1차원 자료로 바껴야댐. 그래서 사용되는것이 Flatten Layer**


입력 이미지가 단채널 3x3 이고 , 2x2 인 필터가 있다

필터 1개면

![](https://github.com/wnsghek31/machine-learning-/blob/master/2x2 1개.PNG)

필터 3개면

![](https://github.com/wnsghek31/machine-learning-/blob/master/2x2 3개.PNG)


필터가 3개라서 출력 이미지도 필터 수에 따라 3개로 늘어났따.
총 가중치 수는 3x2x2 로 12개 (4개씩인 필터가 세개)
**필터마다 고유한 특징을 뽑아 고유한 출력 이미지로 만들기 때문에 필터의 출력값을 더해서 하나의 이미지로 만들거나 그렇게 하지 않는다.** 

이런느김 3개가 나오는것과 같은 이치
![](https://github.com/wnsghek31/machine-learning-/blob/master/필터3개.PNG)


![](https://github.com/wnsghek31/machine-learning-/blob/master/표현1.PNG)


이번에는 입력 이미지의 채널이 3개이고 사이즈가 3x3 , 필터 2x2 1개 사용

![](https://github.com/wnsghek31/machine-learning-/blob/master/채널31.PNG)


필터개수가 3개인것처럼 보이나 이는 입력 이미지에 따라 할당되는 커널이고, 각 커널의 계산 값이 결국 더해져서 출력 이미지 한장을 만들어내므로 필터 개수는 1개이다
> 이는 Dense 레이어에서 입력 뉴런이 늘어나면 거기에 상응하는 시냅스에 늘어나서 가중치의 수가 늘어나는 것과 같은 원리입니다

가중치는 2 x 2 x 3으로 총 12개 이지만 필터 수는 1개

![](https://github.com/wnsghek31/machine-learning-/blob/master/채널32.PNG)

마지막으로 입력 이미지의 사이즈가 3x3 이고 , 채널이 3개이고 , 사이즈가 2x2인 필터가 2개인 경우를 살펴본다.

![](https://github.com/wnsghek31/machine-learning-/blob/master/막1.PNG)

필터가 2개이므로 출력이미지도 2개이다

![](https://github.com/wnsghek31/machine-learning-/blob/master/.PNG)


# 여기까지봣다

***
이부분 헷갈리면 http://umbum.tistory.com/223 가서 보기.

**채널은 하나의 2차원 배열(행렬) 이라고 생각하면 편하다
input은 3개의 channel 이면 , 각각의 channel 당 하나의 필터 channel 이 적용된다. (RGB인 사진은 채널이 R , G , B 로 세개인 것) ==[밑에 출처에서의 예에서 2차원이라 생각하면 편한건가??]==**

즉 **하나의 입력채널에는 하나의 필터채널이 필요**하다
이는 **입력 데이터의 채널 수와 필터 채널 수가 일치해야한다는것**

각각의 필터 채널에서 연산한 값을 모두 더한 값이 output 채널이 된다. 필터 채널이 얼마가 있든 , 결국 다 더해져 하나의 output 채널을 구성함. 따라서 **output 채널의 수는 필터 채널의 수와는 관련없고 , 필터가 몇개있느냐가 결정한다.** 또한 각 필터 채널에서 연산한 값을 모두 더해야 하므로 **모든 채널의 필터는 같은 크기여야한다**

RGB인 사진은 채널이 R , G , B 로 세개인 것이고 , 필터들의 채널은 그 세개인 채널에 맞춰서 세개가 된것이다.  각체낼마다 필터W가 있어서 (W0 ,W1 ,W2) 채널마다 학습을한다!!
**흑백영상인 경우에는 채널이 1이고, 컬러(RGB)영상인 경우에는 채널을 3**

***

### Pooling Layer
여러 풀링 방법이있는데 , 이미지 분야에서는 주로 max pooling을 사용하여 그냥 풀링이라하면 보통 max 풀링이다.

* window size와 stride 사이즈를 같은 값으로해서 모든 원소가 한번씩만 연산에 참여하도록 한다
* 학습해야할 매개변수 x
* 채널 수가 변하지않는다
	> Conv Layer에서는 각 필터 채널을 적용한 결과 채널들을 다 더해야 하나의 output 채널이 되지만 , Pooling layer에서는 결과를 더하지 않는다. 결과 채널이 그대로 output 채널이 되기떄문에 채널수 그대로
* 입력 데이터의 변화에 민감하게 반응하지않음 (Robust)
	> polling layer의 사용 목적이 여기있는데 내가 찾아내고자 하는 특징의 위치를 중요하게 여기기보다는 , input이 그 특징을 포함하고 있느냐 없느냐를 판단하도록 하기 위해서 주변에 있는 값들을 뭉뜽그려서 보겠다는거다.




alexnet에서 두개로 나눠서 처리한것은 GPU 두개 썻어가지고 , 하드웨어에 맞췄떤것

CNN은 대체로 depth가 깊어지는 방향으로 연구를하고 , 거기서 학습이안되는걸 찾아내면서 성능 향상중

[3,3,32,64]
3x3 32개를 받아서 필터를 64개 쓴다?



https://tykimos.github.io/2017/01/27/CNN_Layer_Talk/
http://umbum.tistory.com/227?category=751025
볼것들
http://taewan.kim/post/cnn/
http://hamait.tistory.com/535
