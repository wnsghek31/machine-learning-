

==로지스틱회귀 하나가 퍼셉트론 아닌가?==





Overfitting 문제는 Dropout이나 Batch Normalization같은 것으로 해결
local minima 문제는 backpropagattion이나 데이터와 층을 미리 학습하는 pretraining 방식으로 해결




GD

내가 산의 가장 낮은 위치로 내려가야하는 상황이라면 어떻게 내려가면 낮은 위치에 도달할 수 있을까? 가장 간단한 방법은 가장 기울기가 가파른 방향을 골라서 내려가는 것이다


SGD

SGD는 모든 데이터의 gradient를 평균내어 gradient update를 하는 대신 (이를 ‘full batch’라고 한다), 일부의 데이터로 ‘mini batch’를 형성하여 한 batch에 대한 gradient만을 계산하여 전체 parameter를 update한다



만들고 -> 백프로포게이션 -> RELU




### 커널 트릭
선형분류기인데 곡선으적으로 분류하는 방법.

4차원 featrue에 추가를 x1의 제곱을 넣어준다면 , 판별 함수의 식은 s = XW + b 로 동일하겠지만 , x1^2 에 의해서 이차식 형태를 띄게 된다.
단순히 판별함수에 학습시키는 인풋 데이터를 변형함으로써 곡선적으로 분류하는 선형 분류기를 만들 수 있는 것
인풋 데이터를 일정 규칙에 따라 변형시켜주는 함수를 커널 함수(kernel function)라고 하며 커널 함수를 사용해서 곡선적으로 분류하는 선형분류기를 만드는 것을 ‘커널 트릭(kernel trick)을 사용한다’고 한다.


XOR문제도 다항(polynomail) 커널을 사용해서 feature를 변형하면 분류가 가능해진다.
똑같은 선형 분류기를 사용해도 단순히 인풋 데이터를 변형하기만 해도 몇몇 형태의 곡선적 데이터는 성공적으로 분류할 수 있다.
그러나 커널 트릭은 근본적인 해결법이 되지못한다. 해결할수있는 곡선적 분류문제도 극히일부.
그래서 높은 성능을 내는 비선형 분류기를 구현할 필요가있다.
어떤 형태의 곡선적 또는 비선형 문제든지 모두 학습이 가능하도록 고안된 분류 모형이 **인공신경망(Artifical Neural Network; ANN)**


[사진]

퍼셉트론 하나의 판별함수는  s  = f(XW + b) 처럼 나타내진다.
선형조합 XW+b에 활성화 함수(항상 비선형) f를 씌운것.
**퍼셉트론은 선형 분류기에 비선형 변형을 가한 출력값을 반환한다 (비선형 함수값을 준다).**



SoftMax , CrossEntropy

Softmax(소프트맥스)는 입력받은 값을 출력으로 0~1사이의 값으로 모두 정규화하며 출력 값들의 총합은 항상 1이 되는 특성을 가진 함수

logistic regression 한것도 확률값
sigmoid를 확장시켜 만든 softmax (k=2 이면 sigmoid랑 같은형태)

이 식을 토대로 비용함수를 구하는 방식이 Cross entropy


## 지도학습
비지도학습은 거의 사용하지않는다. 잘되는것은 거의 지도학습이다. 
인공지능회사들 레이블링하는데 사람을 많이쓴다.



## 텐서플로우
수학공식을 그래프로 바꿔서   (Data Flow Graph)
노드는 수학적 연산을 나타내며 , 

tensorflow는
1. build graph using TensorFlow operations (node로?)
2. feed data and run graph (operation , sess.run)
3. update variable in the graph

deep learning에서는 float32 float64 같이 16bit 32bit 64bit의 차이가 중요함
(16bit로 하면 더 빠르니까 , 데이터가많으면 느려서 속도 높이기위해)

reduce_mean : sum 해서 평균

**텐서플로우 워크플로우**

1. 변수와 hypothesis 만들고
2. cost함수 만들고
3. optimizer 를 만들고
4. sess.run 돌리면서 반복

그래픽연산떄문에 sess.tf.Session   sess.run(adding) 으로 코딩해줘야하는것
이 코드들 전까지만 치는것은 그래프 구조를 만들어주는 작업을한것이고 이것으로 그래프를 실행시키는것이다.

`/content/gdrive/My Drive/ `까지는 똑같고   거기서 내가만든 Colab Notebooks에있는 내파일

```
from google.colab import drive
drive.mount('/content/gdrive')

df = pd.read_csv('/content/gdrive/My Drive/Colab Notebooks/DeepLearningZeroToAllExample/Admission_Predict_Ver1.1.csv')
# 구글 드라이브 mount 해주니까 pandas로도 읽어올수잇다.
x = df.iloc[:,1:-1].values
# dataframe에서는 slice를 iloc으로 한다  , [:,:] [행,열]
y = df.iloc[:,[-1]].values
# 밑에다가 sess.run 해서 넣을떄 numpy array 를 넣어야대는데   . .values()하면 array로 바뀐다.
```

placeholder로 선언해주면 아주 flexible하게 만들수있따. 어떤 형태이든 들어갈수있어

we use `tf.Variable` for trainable variables such as weights (W) and biases (B) for your model.
`tf.placeholder` is used to feed actual training examples.

`W = tf.Variable(tf.random_normal([3, 1]), name='weight')`
W의 행의 수는 피쳐의 개수




## 비지도학습
구글의 뉴스 그룹핑
클러스터링



## 뉴럴네트워크


![뉴럴넷](https://github.com/wnsghek31/machine-learning-/blob/master/%EB%89%B4%EB%9F%B4%EB%84%B7.PNG)

![퍼셈트론](https://github.com/wnsghek31/machine-learning-/blob/master/%ED%8D%BC%EC%85%89%ED%8A%B8%EB%A1%A0.PNG)

**입력값들(X)에 가중치를 두어(W) 값(f(x))을 구하고 그 값과 임계치와의 관계를 활성함수(active function)로 판단하여 결과값을 출력하는 것입니다
 : 이때 활성함수는 뉴런에서 임계값을 넘었을 때만 출력하는 부분을 표현한 것**


멀티레이어 퍼셉트론 = 신경망

간단한 분류와 같은 것은 왼쪽 처럼 1층의 뉴런만으로 가능하지만 문제가 복잡할수록 불안정해지고 (xor도 처리 못하는…)
그리고 이렇게 모델을 구성하면 엄청나게 많은 연산을 해야했고… 그 당시 컴퓨터는 절대 불가능


**실제로 알파고는 13개의 층으로 수백 ~ 수천만개의 연결 고리를 갖고 있었다**


크로스엔트로피에서 -log 를 곱하니까 ,  log 형태의 convex한 cost function이 나오게되서 gradient descent 된다.

크로스엔트로피 ,  소프트맥스 다 exponenetial이 있어서 log를 취했을때 convex

sigmoid tanh  같은것이 다 비선형인데 .. costfunction을 계싼하기위해서 log(?)를 취해서(==log를 취하는게 아닌 다른방법도?==)


**분류문제에 있어서는 sigmoid 나 tanh를 취해주니까 비선형 함수가 된다. 그래서 제곱을 취했을때 로컬미니멈 많이생기는 울퉁불퉁한 그래프가...
(분류문제 풀라고 시그모이드 취했떠니 비선형이라(?) 제곱하니까 울퉁불퉁하네?? 펴줘야겟다 -> log)**

멀티레이어 퍼셉트론

 ‘multi layer perceptron (MLP)’라는 구조만 다룰 것인데, 이 구조는 directed simple graph이고, 같은 layer들 안에서는 서로 connection이 없다. 즉, self-loop와 parallel edge가 없고, layer와 layer 사이에만 edge가 존재하며, 서로 인접한 layer끼리만 edge를 가진다. 즉, 첫번째 layer와 네번째 layer를 직접 연결하는 edge가 없는 것이다
 information progation이 ‘forward’로만 일어나기 때문에 이런 네트워크를 feed-forward network라고 부르기도 한다

### back propagiation

![](https://github.com/wnsghek31/machine-learning-/blob/master/%EB%B0%B1%ED%94%84.jpg)

즉, 자칫하면 뉴럴 네트워크의 학습 과정을 추상적으로만 생각하고 지나가버리는 함정에 빠질 수 있다는 것!
실제로 구현해보지 않고 머리로만 이해한 사람은 신경망을 설계할 때, 단순히 layer를 쌓아주기만 하면 역전파(Backprop)가 "magically make them work on your data"라고 믿게 될 수 있으나...


### Activation Function

Activation function은 말 글대로 활성함수로 각 뉴런에 입력된 데이터에 대한 뉴런의 출력이다.
즉 Neural Network에서는 한 노드는 inputs과 각 weights과의 곱을 모두 더한 값을 입력는데, 받아들인 이 값이 Activation function을 지난 값이 이 뉴런이 출력한 output

**미분이 가능해야 gradient descent를 쓰기때문에 미분이 가능한 activation function을 써야댐**


## 깊이가 깊어질때의 문제

잘못된 비선형 함수를 사용해서  결정적으로 성능이 안나왔떤것이다. 

### Vanishing Gradient Problem

backpropagation이 뒤로가면서 학습시키는데 , 뒤쪽으로 갈수록 영향을 못미쳤따. (5~6년전 해결)

**XOR문제인데도 9 depth 까지가면 0.5의 정확도가나온다 2depth만해도 1 나왔엇는데,**

**Back propagation에서 적은 몇개의 hidden layer에서는 되지만 , 여러개 되면못해
뒤에서 error 를 보낼대 이 의미가 갈수록 약해져서서 앞엔 거의 전달 X -> 학습 X**

**이문제의 근본 원인이  sigmoid 활성화 함수 미분의 본질때문이다.**

(https://brunch.co.kr/@chris-song/39)


#### ReLU ( Max(0,x) )
sigmoid function에서의 Gradient Vanishing문제 떄문에 해결법으로써 쓰이기시작함. 
sigmoid fuction으로 인해 0~1의 값을 가지는데, gradient descent를 사용해 backpropagation 수행시 layer를 지나면서 gradient를 계속 곱하므로 gradient는 0으로 수렴하기에 layer가 많아지면 잘작동하지않았떤것

**hidden layer에는 ReLU를 쓰지만 맨 마지막은 sigmoid를 써서 0~1사이 값을 받아야함**

#### 이점
1. Sparse activation : 0 이하의 입력에 대해 0을 출력함으로 부분적으로 활성화 시킬수있따
2. Efficient gradient propagation : gradient의 vanishing이 없으며 gradient가 exploding 되지 않는다
3. Efficient computation : 선형함수이므로 미분 계싼이 매우 간단하다

==시그모이드함수에서의 경사값이아니라, 시그모이드함수에서나온 결과에 코스트함수에서의 경사
그러니까 ReuLU를 써서 정확도가 올라가지 . 이건 음수는 0으로 만들어버리니까  (먼소리지)==



현실적으로 local minimum을 찾는다 . global minimum을 찾을 방법은없다..

이 간단한 gradient descent는 간단할때에만 가능하고 , 
local minimum을 건너뛰고싶은 경우가 있끼때문에 , 더 좋은 방법을 쓴다

출력이 n개인 경우  (출력이 n개인 경우가 현실세계에선 많이있따)
> 게임플레이를 할때 좌표는 x,y로 나타내야하니까


비선형함수인데 제곱을 취하면 로컬미니멈이 엄청나게생김

layer가 한개면 한계가있따. 태생적으로 못푸는문제들이.	


크로스엔트로피에서 -log 를 곱하니까 ,  log 형태의 convex한 cost function이 나오게되서 gradient descent 된다.

크로스엔트로피 ,  소프트맥스 다 exponenetial이 있어서 log를 취했을때 convex


sigmoid tanh  같은것이 다 비선형인데 .. costfunction을 계싼하기위해서 log(?)를 취해서(==log를 취하는게 아닌 다른방법도?==)



데이터 전처리

1. 중심점을 0 가까이로 갖다놓는것.
2. ++표준화(standardization, = normalization 정규분포로 만드는것  )(너무 멀리간 이상값들을 처리할수있따. 영역안으로 넣어서 엄청 큰차이가안나게하는듯?)++
3. ++스케일링 (min-max scale)++

++는 많이 쓰이는것.

데이터가 많은 경우 validation 까지 ,  파라미터 튜닝용으로 . 데이터많으면 학습한번에 몇일씪 걸리니까....


### softmax , CrossEntropy

Softmax function은 Neural network의 최상위층에 사용하여 Classification을 위한 function으로 사용한다. NN의 최상위 계층에 사실 Sigmoid function을 사용하여 실제값과 출력값을 차이를 error function으로 사용해도 된다. 하지만 Softmax function을 사용하는 주요한 이유는 결과를 확률값으로 해석하기 위함( 다른 출력값들과 상대적 비교과 된다. 클래스들 확률 합은 1이 되기에 normalization 효과까지)

※. Softmax function은 logistic regression이 2가지 레이블 분류하는 것과 달리 여러개의 label에 대해서도 분류(multiclas classification)를 가능하게 한다. 2개의 레이블에 대한 분류에 대해 Softmax function과 logistic function은 같다(동치). 



Softmax(소프트맥스)는 입력받은 값을 출력으로 0~1사이의 값으로 모두 정규화하며 출력 값들의 총합은 항상 1이 되는 특성을 가진 함수

logistic regression 한것도 확률값
sigmoid를 확장시켜 만든 softmax (k=2 이면 sigmoid랑 같은형태)

이 식을 토대로 비용함수를 구하는 방식이 Cross entropy


### 초기값설정

초기값이 러닝에 많은 영향을 끼치더라..

초기값을 정하기위해 Restricted Boatman Machine 썼었는데 ,너무 복잡하고 오래걸려서 
`( 입력 개수 , 출력의 개수 중 난수 하나를 뽑아 ) / (입력 개수)의 제곱근?` 으로 Weight을 주니까 학습이 엄청 잘되더라

`( 입력 개수 , 출력의 개수 중 난수 하나를 뽑아 ) / (입력 개수 / 2)의 제곱근?` 으로 하니까 훨씬 더 잘되더라..

### DropOut
오버피팅 방지를위해 Drop out
학습을 진행할때 랜덤으로 몇개에 노드는 학습 시키지않는다 (500개 학습하고 500개는 쉬고 )
이걸 반복해가면서..  [들어오는 데이터에 대해서 어떤 노드는 학습되고 어떤애는 안되고..]
**드롭아웃 비율은 0.5를 많이 준다.**
왜이렇게 되는지에대한 가설(?)의 예로 사진의 종류에따라서 고양이 귀가 가려졌따고 고양이가 아닌게 아닌데 (눈이 두개있어야지 고양이라고하는 모델은 정확한 모델이 아닌것) , 
**dropout은 분류문제에서 많이쓰이지만 회귀문제에선 거의 안쓰여... 몇개뺴고학습하는게 회귀에선 안좋은듯**

#### dropout의 효과
#####  voting 효과
mini-batch 로 줄어든 망으로 학습하게되면 그망은 그망 나름대로 overfitting되고 , 다른 mini-batch 망도 일정정도 overfitting된다 . 이를 무작위 반복하면 voting에의해 평균효과를 얻을수있따.

##### parameter들의  co-adaptation을 피하는 효과
특정 뉴런의 바이어스나 가중치가 큰값을 갖으면 그것의 영향이 커지면서 다른 뉴런들의 학습속도가 느려지거나 학습이 제대로 안됨 하지만 drop out하면 어떤 뉴런의 가중치나 바이어스가 특정 뉴런의 영향을 받지 않기 떄문에 (랜덤하게 선택되기에 어느것은 그 특정 뉴런이 꺼져있으면 영향안받고 어떤건 받게되기에?)굿. 
특정 학습 데이터나 자료에 영향을 받지 않는 보다 robust한 망을 구성

뉴런들을 무작위로 생략시키면서 학습시키면 parameter들이 서로 동화(co-adaptation) 되는것을 막을수있어, 좀더 의미 있는 특징들을 더 추출한다. 즉 다른 파라미터와 같이 cost function을 줄여나가다 보면 파라미터의 공조현상이 일어날수있는데 , Dropout을 하게 되면서 서로 의지하던것을 스스로 해줘야하기때문에 좀더 의미있는 feature를 끄집어낸다

파라미터 0.4 이하에선 underfitting , 0.8 이상에선 overfitting이 되는..
뉴런개수 많으면 , 비율이 0.3 0.4 해도 뉴런개수 적을떄보다 underfitting 피한다 (당연)
0.5가 효과적이다.

DropConnect란 방법도 있는데 거의 비슷한 성능을



depth가 152개가되고 너무 deep해지니까 학습이 잘안되니까 하기위해서 fast foward . weight을 앞으로보내서 더해줌 


