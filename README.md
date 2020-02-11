# 월간 데이콘 1 반도체 박막 두께 분석

---

# 1. 서론

MLP(multi-layer perception, 다층퍼셉트론)는 퍼셉트론으로 이루어진 층(layer) 여러 개를 순차적으로 붙여놓은 형태로써, 그 구조는 아래 그림과 같다.

<img src="https://i.imgur.com/FV67Jyt.png" width="100%">

layer는 크게 Input layer, Hidden layer, Output layer로 구성되어 있고, 각각의 layer는 해당 layer의 input data와 output data의 형태에 알맞게 수많은 weight(가중치)를 가지고 있다. 이 weight들은 gradient descent algorithm의 원리에 의해 train data를 입력시켜줌으로써 업데이트되고, 그 과정 중에 우리가 흔히 알고 있는 propagation과 backpropagation이 적용된다. 그리고 우리는 이 일련의 과정을 'train(학습)'이라고 한다.<br><br>
기존 Machine Learning 모델들과는 달리 MLP는 (물론 중요한 feature만 train시키는 것이 더 효율적이겠지만) input data에서 어떤 feature가 중요한 정보를 가지고 있는지 스스로 찾아내기 때문에 feature의 수에 제한이 없다. 또한 input data의 수가 충분히 많다면 아주 좋은 성능을 낼 수 있는 모델이기도 하다.<br><br>
하지만 이 모델을 사용함에 있어서 직면하는 문제점 또한 중요한 이슈이다. MLP는 말 그대로 multi-layer인데, 이는 다수의 layer를 쌓는 것을 의미한다. 즉 layer의 수에 제한이 없다는 것이다. 이와 같이 모델을 구상하는 과정에 있어서 설계자의 개입이 필요하고 이는 모델의 성능을 좌우하는 중요한 요인이 된다. 왜냐하면 layer의 수가 너무 많으면 모델의 복잡성이 증가하여 train set은 잘 맞추지만 test set은 잘 맞추지 못하는 overfitting이 발생하기 쉽고, 반대로 layer의 수가 너무 적으면 모델이 너무 단순하여 train set의 특징을 weight에 모두 반영하지 못해 train set과 test set 모두 잘 맞추지 못하는 underfitting이 발생하기 쉽기 때문이다. layer의 수 뿐만 아니라, 각 layer안의 weight의 개수, layer와 layer사이에 통과하는 activation function의 종류, 학습에 적용하는 optimizer의 종류, learning rate, batch size의 크기 등 너무도 많은 하이퍼파라미터들이 존재한다. 이 외에도 특정 상황에 따라 모델의 성능을 향상시키는 Regularization(L1, L2), Dropout, weight initialization, BatchNormalization 등 다양한 기법들이 존재한다.<br><br>
따라서 이번 Competition을 통해 MLP를 직접 구상해봄으로써 기존 Machine Learning 모델들과의 성능을 비교해보고 layer의 수나 weight의 수 등 모델을 구상하는 데에 있어서 결정해야 하는 중요한 요소들, 그리고 하이퍼파라미터를 결정하는 데 있어서 발생하는 문제들을 직면하고 해결해나가는 것을 중점으로 한다.

# 2. 대회

이번에 참가하는 Competition은 '월간 데이콘 1 반도체 박막 두께 분석'이다. 이는 데이터 사이언스를 공부하는 사람들에게 잘 알려진 Kaggle과 유사한 DACON에서 개최하였으며, Competition 주소와 대회에 대한 설명은 아래 링크를 통해 볼 수 있다.<br><br>
Competiton 선정은 Deep Learning Method를 사용하기에 적합한 dataset이기에 이것으로 선정했으며, 현 시점 진행 중인 AI Study를 통해 스터디원들과 함께 결정했다.

* Competition 주소<br>
https://dacon.io/competitions/official/235554/overview/description/<br>
* 대회안내<br>
https://github.com/YoonSungLee/DACON-semiconductor-competition-private/blob/master/Competition%20Introduction.ipynb
* AI Study 주소<br>
https://github.com/Inha-AI/DACON-semiconductor-competition

# 3. 연구

[여기를 참조하세요](https://github.com/YoonSungLee/DACON-semiconductor-competition-private/blob/master/DACON%20Semiconductor%20Competition%20Report(Final).ipynb)

# 4. 결과

최종 결과 제출횟수 18회, 점수 0.6181722715점, 등수 26등을 기록했다. 이 정보를 확인할 수 있는 사이트는 아래와 같다.

https://dacon.io/competitions/official/235554/leaderboard/

# 5. 고찰

AI Study를 모집하여 개념에 대한 공부를 하다가 이번에 처음으로 Deep Learning 분야에 대한 Competition에 참가했다. 처음인만큼 모델링을 하면서 실수도 많았고 체계적이지 못한 접근법도 많았다. 하지만 또 반대로 지금까지 공부했던 내용들을 실제로 적용해보면서 이론상으로만 알고 있던 지식들을 몸소 체감하는 기회를 가질 수 있었다. 이번 Competition을 준비하면서 경험했던 특별한 개념들, 실수들을 정립하고 스스로를 피드백함으로써 역량을 키우고자 한다.

#### MLP에 의존한 모델링
<br>
처음 모델링을 시작한만큼 딥러닝 프레임워크 사용법에 대한 공부가 필요했기에 Keras를 선택해 속성으로 공부했다. 여러 모델을 적용하기에 Keras에 대한 숙지가 충분하지 않아서 결국 MLP 모델의 성능을 올리는 데 주력했다. 하지만 이번 Competition은 데이터 사이에 상관관계가 높은 특성을 가지고 있기에 MLP보다는 CNN이나 RNN(또는 LSTM)이 적합하다고 생각한다. 특히 스터디원 중 CNN 모델을 구성한 인원은 상위 등수를 받을 수 있었다. Keras 사용법을 숙지해서 MLP, CNN, RNN 등의 모델에 대하여 기초 모델링을 준비해놓을 필요가 있다. 또한 이번 데이터는 Deep Learning에 적합한 데이터이지만 Machine Learning 모델을 적용해보는 것도 하나의 방안이 될 수 있을 것이다.

#### summary, callback 등 주요한 Keras 도구 숙지
<br>
모델의 아키텍처를 확인하는 방법으로 SVG와 model_to_dot을 사용했는데 적절치 못하다는 판단을 내렸다. 이미지의 사이즈가 너무 커서 모델의 전체적인 모습을 확인하지 못했고, 그 외의 정보들 또한 얻기가 어려웠다. Keras에서는 model.summary의 간단한 코드로 모델의 아키텍처, weight의 수 등을 보기 쉽게 정리해준다. 또한 callback함수를 뒤늦게 알게 되었는데, 이를 통해 overfitting이 발생하기 전까지 학습한 모델과 그 weight를 저장할 수 있다. 이렇게 Keras는 사용자가 원하는 것들을 간편하게 구현할 수 있도록 해놓았다. 이런 부분에 대한 숙지가 필요하다.

#### 체계적이지 않은 모델링과 하이퍼파라미터 탐색 방법
<br>
위의 내용을 보면 알 수 있듯이 굉장히 많은 모델링을 했다. 하지만 자세히 보면 체계적이지 않은 모델링과 하이퍼파라미터 탐색 방법 때문이다. 모델링을 함수를 통해 정의하는 것, 앞으로 모델링해야할 목록들을 구상해놓고 하나씩 소거하는 것, Grid Search, Random Search, BayesianOptimization Search 등 하이퍼파라미터 탐색 방법을 숙지할 필요가 있다. 특히 모델마다 적합한 하이퍼파라미터가 다르기 때문에 모델이 바뀌면 다시 하이퍼파라미터를 탐색해야 한다는 것을 인지할 필요가 있다.
