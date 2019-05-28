# AutoKeras

## Auto-ML

데이터 탐색및 정제에 시간을 많이 쏟기때문에 모델 튜닝이나 정확도 개선을 위한 작업 시간이 부족하다. 여기서 Auto ML의 아이디어가 나왔는데 , 다양한 알고리즘 및 다양한 하이퍼 파라미터를 구성하여 수많은 모델을 생성하고 성능과 정확도를 비교하는 것을 자동화 하는것

데이터의 ‘특성’에 따라 선택하면 좋을 알고리즘과 어울리는 파라미터들을 ‘합리적’으로 ‘찾아 주는 것‘

Machine Learning 으로 설계하는 Machine Learning

### 3가지 

#### Automated Feature Learning
 사람이 직접 실험적으로 feature 추출 방법을 정하는 대신 최적의 feature 추출 방법을 학습을 통해서 찾는것

#### Architecture Search
 ResNet, DenseNet 등 CNN과 LSTM, GRU 등 RNN을 구성하는 network 구조, 즉 architecture를 사람이 직접 하나하나 설계하는 대신 학습을 통해 최적의 architecture를 설계하는 방법
 

#### Hyperparameter Optimization
학습을 시키기 위해 필요한 hyperparameter들을 학습을 통해 추정하는 것을 의미


#### 이러한 작업을 위한 라이브러리는 

* Auto-sklearn
* TPOT
* Auto-Keras
* H2O.ai
* Google's AutoML


## Auto-Keras    

* Auto-Keras는 딥러닝모델의 아키텍처 및 하이퍼 파라미터를 자동으로 검색하는 기능 을 제공

* Bayesian Optimization 알고리즘 기반의 프레임워크 . 강화학습이나 유전 알고리즘 기반 보다 적은 시간과 자원으로 가능.

* AutoKeras 진행 과정을 보면 Father Model을 두고 거기에 added_operation을 적용해 모델 정확도를 높여가는 방식.  딥러닝 네트워크 구조를 바꾼다거나, deeper / wider / concat 등으로 모델에 변형을 준다



> Imageclassifier (CNN)으로 MNIST 15번까지 돌리는데 12시간 이상 소요)  
	== 스펙이 얼마나 되는지 모름 == 
    MNIST 데이터를 이용하여 끝까지 돌리면 0.998%의 높은 정확도가 나온다
	

AutoKeras 라이브러리의 기반이 되는 “Efficient Neural Architecture Search with Network Morphism” 논문

### 오토케라스 구성요소

* 여러개의 분류기 모델이 있는 탐색 공간 : classifier
* 이데아 공간을 탐색하고 최적 모델 구조를 찾아가는 옵티마이저 역할 : searcher
* 이데아 공간의 한 포인트에 대응되는 임의의 모델 구조 : graph
* 고정된 graph를 가지고 입력데이터에 적합하게 파라미터를 학습하는 실제 분류 모델 : model


### 주의

Module 과 Classifier 에서는 쓸수있는 메서드들이 다른거같다
> MlpModule 에서는 load_searcher , load_best_model 이 안됨 .ImageClassifier에서 확인은 못해봄) 



## 과정

### 1. 데이터 불러오기

* **images and the labels are formatted into numpy arrays,**

* channel 까지 있어야함 , (50000,24,24,1) 같이

> 그래서 cifar은 load_data하고 바로 넣는데 MNIST는 reshape로 (60000,24,24) -> (60000,24,24,1) 으로 바꿔줌


```
from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
```
여기서 x  x_train , y_train shape가 (50000,32,32,3) , (50000,1)
` [ [6],[9] ...[1]]`


```
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape + (1,)) # (1,) denotes the channles which is 1 in this case
x_test = x_test.reshape(x_test.shape + (1,)) # (1,) denotes the channles which is 1 in this case
```



### 2. classifier 정의와 학습

* 산책해야할 공간을 정의하는것. ImageClassifier , TextClassifier , MLPModule 등을 정의 

* clf.fit()을 하면 , 탐색 공간에서 최적 graph를 찾도록 searcher가 clf 공간을 탐색

* searcher가 하는일은 공간 내 한 포인트에 대응되는 그래프를 그리고 , 그 그래프로 모델 파라미터를 학습시킨다. 이때 모델을 얼마나 학습시킬지 , 얼마나 학습시켜보고 이그래프를 평가할지를 결정 할수있다.

> default 값은 MAX_ITER_NUM = 200 (epoch 수) , MIN_LOSS_DEC = 1e-4 , MAX_NO_IMPROVEMENT_NUM = 5
	
     
```
clf = ImageClassifier(verbose=True, path='auto-keras/', searcher_args={'trainer_args':{'max_iter_num':5}})
# max_iter_num 은 도는 epoch 수

clf.fit(x_train, y_train, time_limit = 12 * 60 * 60) 
#  모델을 학습하는 시간이 time_limit , 1*1*60 하면 60초
```
### 3. clf의 final_fit

* searcher가 주어진 시간동안 탐색 후 종료하면 탐색한 모델들 중 accuracy 기준으로 best model을 기록해 둔다

* searcher의 history attribute로 탐색한 결과를 한눈에 볼수있다.

```
clf.get_best_model_id()

searcher = clf.load_searcher()
searcher.history
```

* final_fit은 지금 까지 찾은 best model 구조로 최종적으로 모델 파라미터를 학습하게 된다. 여기서도 max_iter_num을 줄수있따

```
clf.final_fit(x_train, y_train, x_test, y_test, retrain=False, trainer_args={'max_iter_num': 10})
```

### 4. 모델 저장하기

* classifier를 불러올때  파라미터 path = "./AutoML" 같이 하면 , AutoML 폴더에 각 모델의 graph 파일이 저장된다.

* 밑에 방법으로 할 수도 있다고한다

```
clf.export_keras_model(model_path_file_name)
```

== clf.save_searcher(searcher) 이거는 안된다고 함  에러남==

### 4. 모델 불러와서 torch model로 변환하기

```
from autokeras.utils import pickle_from_file 
model = pickle_from_file(PATH)
torch_model = model.produce_model()

torch_model

```

```
케라스

keras_model = graph.produce_keras_model()
keras_model.summary()
```


### 5. 모델 시각화

* pip install graphviz 을 해놔야함

* model training 이 끝난 후  밑의 코드를 돌리면 시각화가 된다.


**examples/visualize.py**
```

import os
from graphviz import Digraph

from autokeras.utils import pickle_from_file


def to_pdf(graph, path):
    dot = Digraph(comment='The Round Table')

    for index, node in enumerate(graph.node_list):
        dot.node(str(index), str(node.shape))

    for u in range(graph.n_nodes):
        for v, layer_id in graph.adj_list[u]:
            dot.edge(str(u), str(v), str(graph.layer_list[layer_id]))

    dot.render(path)


def visualize(path):
    cnn_module = pickle_from_file(os.path.join(path, 'module'))
    cnn_module.searcher.path = path
    for item in cnn_module.searcher.history:
        model_id = item['model_id']
        graph = cnn_module.searcher.load_model_by_id(model_id)
        to_pdf(graph, os.path.join(path, str(model_id)))


if __name__ == '__main__':
    visualize('~/asd/')
    # 이 부분에 ImageClassifier(path="~/asd/") 같이 해놨으면 위에 그 값을 넣으면 된다
```


### 6. torch 또는 keras 모델로 변환하기

* searcher의 load_best_model()을 이용해 탐색한 모델 중 가장 좋은 모델을 가져올수있습니다.

* 모델 아키텍쳐(그래프)를 먼저 불러오고, graph의 produce model 메소드를 이용해 토치 모델로 변할수 있습니다. (produce_keras_model 로 변환)

* 해당 메소드는 github repo를 통해서 설치한 경우만 이용가능한다 (18.08.20) << 확인해보자

```
graph = searcher.load_best_model()
# graph = searcher.load_model_by_id(16)

torch_model = graph.produce_model()

torch model
```

### 7. 그래프에서 변환된 모델 학습시키기

* 변환된 torch 또는 kears 모델은 모델 파라미터가 랜덤 초기화 된 모델로 아직 학습 되기 전

* 변환후 모델 학습 과정은 기본 torch 또는 kears 모델 학습과 동일

* 학습 전에 y값 형태를 one-hot-encoding 형태로 변환해줘야함

* 학습 후 모델에 레이어 추가나 feature extractor 역할만 하도록 일부 레이어만 사용하도록 하는 작업이 가능


https://github.com/yjucho1/articles/tree/master/auto-keras



***************************

### 모델 

#### MlpModule (MLP)

MlpModule에서는 train_data  , test_data 들어가는 것들은 모두 dataloader 인듯? , Imageclassifier에서는 그냥 x_train들로 그냥 넣어서함.


`mlpModule = MlpModule(loss, metric, searcher_args, path, verbose)`

* loss와 metric 결정 

* path는 searching process와 model 저장 될 위치

* searcher_args 에 max_iter_num을 줘서 한 모델당 epoch를 제한 할 수있다. max_no_improvement_num는  after max_no_improvement_num, if the model still makes no improvement, finish training.

* verbose는 search 과정이 stdout을 보이게할거냐 말거냐 

`mlpModule.fit(n_output_node, input_shape, train_data, test_data, time_limit=24 * 60 * 60)`

* n_output_node 는 target class 개수  , 
* input shape 은 x_train의 shape 값 (x_train.shape)
* time_limit 은 시간 제한 , 24*60*60 하면 model search를 24시간동안 돌리겠다. 

`mlpModule.final_fit(train_data, test_data, trainer_args=None, retrain=False)`

* retrain: model의 weight 을 reinitialize 할거냐 말거냐
* trainer_args: A dictionary containing the parameters of the ModelTrainer constructor

[https://github.com/keras-team/autokeras/blob/master/examples/net_modules/mlp_module.py]


#### CNNModule (CNN)

`cnnModule = CnnModule(loss=classification_loss, metric=Accuracy, searcher_args={}, path=TEST_FOLDER, verbose=False)	`

`cnnModule.fit(n_output_node, input_shape, train_data, test_data, time_limit=24 * 60 * 60)`

`cnnModule.final_fit(train_data, test_data, trainer_args=None, retrain=False)`


[https://github.com/keras-team/autokeras/blob/master/examples/net_modules/cnn_module.py]


#### TaskModule

##### Automated text classifier tutorial.
https://github.com/keras-team/autokeras/blob/master/examples/task_modules/text/text_classification.py

#### Pretrained model

##### Object detection tutorial.
https://github.com/keras-team/autokeras/blob/master/examples/pretrained_models/object_detection/object_detection_example.py

##### Sentiment Analysis tutorial.
깃헙 에러 

##### Topic Classification tutorial.
깃헙 에러



### 도커

1. docker pull garawalid/autokeras:latest

2. docker run -it --shm-size 2G garawalid/autokeras /bin/bash






cifar은 2018 11월 16일에 error rate 1까찌 떨어졋는데
(https://en.wikipedia.org/wiki/CIFAR-10)
best model이 0.7437 , max_iter_num을 작게줘서 그런듯




conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

