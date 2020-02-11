{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 월간 데이콘 1 반도체 박막 두께 분석"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "---"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 1. 서론"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "MLP(multi-layer perception, 다층퍼셉트론)는 퍼셉트론으로 이루어진 층(layer) 여러 개를 순차적으로 붙여놓은 형태로써, 그 구조는 아래 그림과 같다."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<img src=\"https://i.imgur.com/FV67Jyt.png\" width=\"100%\">"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "layer는 크게 Input layer, Hidden layer, Output layer로 구성되어 있고, 각각의 layer는 해당 layer의 input data와 output data의 형태에 알맞게 수많은 weight(가중치)를 가지고 있다. 이 weight들은 gradient descent algorithm의 원리에 의해 train data를 입력시켜줌으로써 업데이트되고, 그 과정 중에 우리가 흔히 알고 있는 propagation과 backpropagation이 적용된다. 그리고 우리는 이 일련의 과정을 'train(학습)'이라고 한다.<br><br>\n",
    "기존 Machine Learning 모델들과는 달리 MLP는 (물론 중요한 feature만 train시키는 것이 더 효율적이겠지만) input data에서 어떤 feature가 중요한 정보를 가지고 있는지 스스로 찾아내기 때문에 feature의 수에 제한이 없다. 또한 input data의 수가 충분히 많다면 아주 좋은 성능을 낼 수 있는 모델이기도 하다.<br><br>\n",
    "하지만 이 모델을 사용함에 있어서 직면하는 문제점 또한 중요한 이슈이다. MLP는 말 그대로 multi-layer인데, 이는 다수의 layer를 쌓는 것을 의미한다. 즉 layer의 수에 제한이 없다는 것이다. 이와 같이 모델을 구상하는 과정에 있어서 설계자의 개입이 필요하고 이는 모델의 성능을 좌우하는 중요한 요인이 된다. 왜냐하면 layer의 수가 너무 많으면 모델의 복잡성이 증가하여 train set은 잘 맞추지만 test set은 잘 맞추지 못하는 overfitting이 발생하기 쉽고, 반대로 layer의 수가 너무 적으면 모델이 너무 단순하여 train set의 특징을 weight에 모두 반영하지 못해 train set과 test set 모두 잘 맞추지 못하는 underfitting이 발생하기 쉽기 때문이다. layer의 수 뿐만 아니라, 각 layer안의 weight의 개수, layer와 layer사이에 통과하는 activation function의 종류, 학습에 적용하는 optimizer의 종류, learning rate, batch size의 크기 등 너무도 많은 하이퍼파라미터들이 존재한다. 이 외에도 특정 상황에 따라 모델의 성능을 향상시키는 Regularization(L1, L2), Dropout, weight initialization, BatchNormalization 등 다양한 기법들이 존재한다.<br><br>\n",
    "따라서 이번 Competition을 통해 MLP를 직접 구상해봄으로써 기존 Machine Learning 모델들과의 성능을 비교해보고 layer의 수나 weight의 수 등 모델을 구상하는 데에 있어서 결정해야 하는 중요한 요소들, 그리고 하이퍼파라미터를 결정하는 데 있어서 발생하는 문제들을 직면하고 해결해나가는 것을 중점으로 한다."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 2. 대회"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "이번에 참가하는 Competition은 '월간 데이콘 1 반도체 박막 두께 분석'이다. 이는 데이터 사이언스를 공부하는 사람들에게 잘 알려진 Kaggle과 유사한 DACON에서 개최하였으며, Competition 주소와 대회에 대한 설명은 아래 링크를 통해 볼 수 있다.<br><br>\n",
    "Competiton 선정은 Deep Learning Method를 사용하기에 적합한 dataset이기에 이것으로 선정했으며, 현 시점 진행 중인 AI Study를 통해 스터디원들과 함께 결정했다."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* Competition 주소<br>\n",
    "https://dacon.io/competitions/official/235554/overview/description/<br>\n",
    "* 대회안내<br>\n",
    "https://github.com/YoonSungLee/DACON-semiconductor-competition-private/blob/master/Competition%20Introduction.ipynb\n",
    "* AI Study 주소<br>\n",
    "https://github.com/Inha-AI/DACON-semiconductor-competition"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 3. 연구"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### [Baseline Code](https://github.com/YoonSungLee/DACON-semiconductor-competition-private/blob/master/submission_01.ipynb)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Score : 53.3163299561"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* 4 layers\n",
    "* 160 units, relu\n",
    "* Adam\n",
    "* epochs 20\n",
    "* batch_size 10000"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "```\n",
    "# 케라스를 통해 모델 생성을 시작합니다.\n",
    "\n",
    "model = Sequential()\n",
    "model.add(Dense(units=160, activation='relu', input_dim=226))\n",
    "model.add(Dense(units=160, activation='relu'))\n",
    "model.add(Dense(units=160, activation='relu'))\n",
    "model.add(Dense(units=4, activation='linear'))\n",
    "```"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "```\n",
    "# 모델을 컴파일합니다.\n",
    "\n",
    "model.compile(loss='mae', optimizer='adam', metrics=['mae'])\n",
    "```"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "```\n",
    "# 모델을 학습합니다.\n",
    "\n",
    "model.fit(train_X, train_Y, epochs=20, batch_size=10000, validation_split=0.05)\n",
    "```"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### [Model 1](https://github.com/YoonSungLee/DACON-semiconductor-competition-private/blob/master/submission_01.ipynb)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Score : 8.4483604431"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* 5 layers\n",
    "* 160 units, he_normal, relu\n",
    "* BatchNormalization\n",
    "* Adam(0.001)\n",
    "* epochs 50\n",
    "* batch_size 1000"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "```\n",
    "# 케라스를 통해 모델 생성을 시작합니다.\n",
    "\n",
    "model_t1 = Sequential()\n",
    "model_t1.add(Dense(units=160, input_dim=226, kernel_initializer='he_normal'))\n",
    "model_t1.add(BatchNormalization())\n",
    "model_t1.add(Activation('relu'))\n",
    "model_t1.add(Dense(units=160, kernel_initializer='he_normal'))\n",
    "model_t1.add(BatchNormalization())\n",
    "model_t1.add(Activation('relu'))\n",
    "model_t1.add(Dense(units=160, kernel_initializer='he_normal'))\n",
    "model_t1.add(BatchNormalization())\n",
    "model_t1.add(Activation('relu'))\n",
    "model_t1.add(Dense(units=160, kernel_initializer='he_normal'))\n",
    "model_t1.add(BatchNormalization())\n",
    "model_t1.add(Activation('relu'))\n",
    "model_t1.add(Dense(units=4, activation='linear'))\n",
    "\n",
    "adam = keras.optimizers.Adam(0.001)\n",
    "model_t1.compile(loss='mae', optimizer=adam, metrics=['mae'])\n",
    "\n",
    "model_t1.fit(train_X, train_Y, epochs=50, batch_size=1000, validation_split=0.05)\n",
    "```"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* layer 수를 늘려 모델의 복잡성을 증가시켰다.\n",
    "* weight의 초기값은 Activation function을 'relu' 함수로 사용하고 있기 때문에 'he_normal'로 설정했다.\n",
    "* epoch 수를 늘리고 batch_size는 줄여서 가중치 업데이트량을 최대한 증가시켰다.\n",
    "* 각 layer와 activation function 사이에 BatchNormalization을 배치시켰는데, 이를 통해 오버피팅 문제를 초기에 해결할 수 있다. 또한 이 방법을 사용하면 weight의 초깃값 설정에 모델의 성능이 크게 좌우되지 않는다는 것을 경험적으로 알 수 있었다."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### [Model 2](https://github.com/YoonSungLee/DACON-semiconductor-competition-private/blob/master/submission_01.ipynb)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Score : 6.0977301598"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* 5 layers\n",
    "* 160 units, he_normal, relu\n",
    "* BatchNormalization\n",
    "* Adam(0.001)\n",
    "* epochs 100\n",
    "* batch_size 500"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "```\n",
    "# 케라스를 통해 모델 생성을 시작합니다.\n",
    "\n",
    "model_t2 = Sequential()\n",
    "model_t2.add(Dense(units=160, input_dim=226, kernel_initializer='he_normal'))\n",
    "model_t2.add(BatchNormalization())\n",
    "model_t2.add(Activation('relu'))\n",
    "model_t2.add(Dense(units=160, kernel_initializer='he_normal'))\n",
    "model_t2.add(BatchNormalization())\n",
    "model_t2.add(Activation('relu'))\n",
    "model_t2.add(Dense(units=160, kernel_initializer='he_normal'))\n",
    "model_t2.add(BatchNormalization())\n",
    "model_t2.add(Activation('relu'))\n",
    "model_t2.add(Dense(units=160, kernel_initializer='he_normal'))\n",
    "model_t2.add(BatchNormalization())\n",
    "model_t2.add(Activation('relu'))\n",
    "model_t2.add(Dense(units=4, activation='linear'))\n",
    "\n",
    "adam = keras.optimizers.Adam(0.001)\n",
    "model_t2.compile(loss='mae', optimizer=adam, metrics=['mae'])\n",
    "\n",
    "model_t2.fit(train_X, train_Y, epochs=100, batch_size=500, validation_split=0.05)\n",
    "```"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* Model 1의 결과를 통해 아직 학습이 덜 된 상태라고 판단하여 epochs을 더욱 늘리고 batch_size는 더욱 줄여서 가중치 업데이트량을 증가시켰다.\n",
    "* 이렇게 설정하면 Model 1과 비교했을 때 모델의 성능이 소폭 향상했음을 알 수 있다."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### [Model 3](https://github.com/YoonSungLee/DACON-semiconductor-competition-private/blob/master/submission_03.ipynb)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Score : 19.5661201477"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* 5 layers\n",
    "* 160 units, he_normal, relu\n",
    "* BatchNormalization\n",
    "* Dropout\n",
    "* Adam(0.001)\n",
    "* epochs 100\n",
    "* batch_size 500"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "```\n",
    "# 케라스를 통해 모델 생성을 시작합니다.\n",
    "\n",
    "model_03 = Sequential()\n",
    "model_03.add(Dense(units=160, input_dim=226, kernel_initializer='he_normal'))\n",
    "model_03.add(BatchNormalization())\n",
    "model_03.add(Activation('relu'))\n",
    "model_03.add(Dropout(0.3))\n",
    "model_03.add(Dense(units=160, kernel_initializer='he_normal'))\n",
    "model_03.add(BatchNormalization())\n",
    "model_03.add(Activation('relu'))\n",
    "model_03.add(Dropout(0.3))\n",
    "model_03.add(Dense(units=160, kernel_initializer='he_normal'))\n",
    "model_03.add(BatchNormalization())\n",
    "model_03.add(Activation('relu'))\n",
    "model_03.add(Dropout(0.3))\n",
    "model_03.add(Dense(units=160, kernel_initializer='he_normal'))\n",
    "model_03.add(BatchNormalization())\n",
    "model_03.add(Activation('relu'))\n",
    "model_03.add(Dropout(0.3))\n",
    "model_03.add(Dense(units=4, activation='linear'))\n",
    "\n",
    "adam = keras.optimizers.Adam(0.001)\n",
    "model_03.compile(loss='mae', optimizer=adam, metrics=['mae'])\n",
    "\n",
    "model_03.fit(train_X, train_Y, epochs=100, batch_size=500, validation_split=0.05)\n",
    "```"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* Model2의 loss와 val_loss의 차이가 너무 심하여 이를 Overfitting 현상이라고 판단하고(후에 잘못된 모델 설정 때문에 사실 Overfitting이 아님이 밝혀진다) Dropout 방법을 이용했다. 통상적으로 rate를 0.5로 시작한다고 알려져있지만, 먼저 0.3으로 설정했다.\n",
    "* 결과는 대실패다. 이전 모델에 비해 성능이 오히려 하락했기 때문에, rate 비율을 바꾸거나 다른 방법 도입이 필요했다."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### [Model 4](https://github.com/YoonSungLee/DACON-semiconductor-competition-private/blob/master/submission_04.ipynb)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Score : 3.347700119"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* 6 layers\n",
    "* (239, 252, 265, 178, 91, 4) units, he_normal, relu\n",
    "* BatchNormalization\n",
    "* Adam(0.001)\n",
    "* epochs 100\n",
    "* batch_size 100"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "```\n",
    "# 케라스를 통해 모델 생성을 시작합니다.\n",
    "\n",
    "model_04 = Sequential()\n",
    "model_04.add(Dense(units= 239, input_dim=226, kernel_initializer='he_normal'))\n",
    "model_04.add(BatchNormalization())\n",
    "model_04.add(Activation('relu'))\n",
    "model_04.add(Dense(units=252, kernel_initializer='he_normal'))\n",
    "model_04.add(BatchNormalization())\n",
    "model_04.add(Activation('relu'))\n",
    "model_04.add(Dense(units=265, kernel_initializer='he_normal'))\n",
    "model_04.add(BatchNormalization())\n",
    "model_04.add(Activation('relu'))\n",
    "model_04.add(Dense(units=178, kernel_initializer='he_normal'))\n",
    "model_04.add(BatchNormalization())\n",
    "model_04.add(Activation('relu'))\n",
    "model_04.add(Dense(units=91, kernel_initializer='he_normal'))\n",
    "model_04.add(BatchNormalization())\n",
    "model_04.add(Activation('relu'))\n",
    "model_04.add(Dense(units=4, activation='linear'))\n",
    "\n",
    "adam = keras.optimizers.Adam(0.001)\n",
    "model_04.compile(loss='mae', optimizer=adam, metrics=['mae'])\n",
    "\n",
    "model_04.fit(train_X, train_Y, epochs=100, batch_size=100, validation_split=0.05)\n",
    "```"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* layer의 층을 더 쌓고, 양 쪽 layer units의 평균을 해당 layer units으로 설정했다(이는 google search를 통해 한 데이터사이언티스트의 경험에 의거했다).\n",
    "* 예를 들어 양 쪽 layer의 units을 각각 200, 400으로 설정했다면 가운데 layer의 units은 300으로 설정하는 방식이다. 우리는 input data의 feature의 수와 output data의 feature의 수를 알고 있기 때문에, 모든 layer의 units을 변수로 두고 방정식을 통해 이 값들을 이끌어 낼 수 있다.\n",
    "* batch_size를 대폭 줄여서 가중치의 업데이트량을 증가시켰다."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### [Model 5](https://github.com/YoonSungLee/DACON-semiconductor-competition-private/blob/master/submission_05.ipynb)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Score : 3.0454199314"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* 6 layers\n",
    "* (239, 252, 265, 178, 91, 4) units, he_normal, relu\n",
    "* BatchNormalization\n",
    "* Adam(0.001)\n",
    "* epochs 100\n",
    "* batch_size 500"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "```\n",
    "# 케라스를 통해 모델 생성을 시작합니다.\n",
    "\n",
    "model_05 = Sequential()\n",
    "model_05.add(Dense(units= 239, input_dim=226, kernel_initializer='he_normal'))\n",
    "model_05.add(BatchNormalization())\n",
    "model_05.add(Activation('relu'))\n",
    "model_05.add(Dense(units=252, kernel_initializer='he_normal'))\n",
    "model_05.add(BatchNormalization())\n",
    "model_05.add(Activation('relu'))\n",
    "model_05.add(Dense(units=265, kernel_initializer='he_normal'))\n",
    "model_05.add(BatchNormalization())\n",
    "model_05.add(Activation('relu'))\n",
    "model_05.add(Dense(units=178, kernel_initializer='he_normal'))\n",
    "model_05.add(BatchNormalization())\n",
    "model_05.add(Activation('relu'))\n",
    "model_05.add(Dense(units=91, kernel_initializer='he_normal'))\n",
    "model_05.add(BatchNormalization())\n",
    "model_05.add(Activation('relu'))\n",
    "model_05.add(Dense(units=4, activation='linear'))\n",
    "\n",
    "adam = keras.optimizers.Adam(0.001)\n",
    "model_05.compile(loss='mae', optimizer=adam, metrics=['mae'])\n",
    "\n",
    "hist = model_05.fit(train_X, train_Y, epochs=100, batch_size=500, validation_split=0.05)\n",
    "```"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<img src=\"https:/i.imgur.com/DItRXPh.png\" width=\"100%\">"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* batch_size를 증가시켰더니 오히려 성능이 향상되었다. 이를 통해 batch_size를 줄여 weight의 업데이트량을 늘리는 것이 무조건적으로 좋은 방법은 아니라는 것을 확인할 수 있다. 적절한 batch_size 설정이 중요하다.\n",
    "* 위의 그래프는 학습곡선을 나타내는데, metrics 설정을 잘못하여 accuracy가 loss와 동일하게 나온다. 차후 모델에서 수정한 결과를 확인할 수 있다."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### [Model 6](https://github.com/YoonSungLee/DACON-semiconductor-competition-private/blob/master/submission_06.ipynb)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Score : 5.5031199455"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* 5 layers\n",
    "* 178 units, he_normal, relu\n",
    "* BatchNormalization\n",
    "* Adam(0.001)\n",
    "* epochs 100\n",
    "* batch_size 500"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "```\n",
    "# 케라스를 통해 모델 생성을 시작합니다.\n",
    "\n",
    "model_06 = Sequential()\n",
    "model_06.add(Dense(units= 178, input_dim=226, kernel_initializer='he_normal'))\n",
    "model_06.add(BatchNormalization())\n",
    "model_06.add(Activation('relu'))\n",
    "model_06.add(Dense(units=178, kernel_initializer='he_normal'))\n",
    "model_06.add(BatchNormalization())\n",
    "model_06.add(Activation('relu'))\n",
    "model_06.add(Dense(units=178, kernel_initializer='he_normal'))\n",
    "model_06.add(BatchNormalization())\n",
    "model_06.add(Activation('relu'))\n",
    "model_06.add(Dense(units=178, kernel_initializer='he_normal'))\n",
    "model_06.add(BatchNormalization())\n",
    "model_06.add(Activation('relu'))\n",
    "model_06.add(Dense(units=4, activation='linear'))\n",
    "\n",
    "adam = keras.optimizers.Adam(0.001)\n",
    "model_06.compile(loss='mae', optimizer=adam, metrics=['mae'])\n",
    "\n",
    "hist = model_06.fit(train_X, train_Y, epochs=100, batch_size=500, validation_split=0.05)\n",
    "```"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<img src=\"https://i.imgur.com/zn4J8rC.png\" width=\"100%\">"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* 이번 모델에서는 layer의 층을 줄이고 모든 layer의 units을 일정한 값으로 설정하여 시험했다.\n",
    "* layer의 층이 줄어든 이유인지, 아니면 이전 모델의 units 설정이 좋은건지는 모르겠지만 이번 모델은 저번 모델보다 성능이 떨어졌다.\n",
    "* 위의 그래프는 학습곡선을 나타내는데, metrics 설정을 잘못하여 accuracy가 loss와 동일하게 나온다. 차후 모델에서 수정한 결과를 확인할 수 있다."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### [Model 7](https://github.com/YoonSungLee/DACON-semiconductor-competition-private/blob/master/submission_07.ipynb)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* 8 layers\n",
    "* (108, 82, 56, 30) units, he_normal, relu\n",
    "* BatchNormalization\n",
    "* Adam(0.008)\n",
    "* epochs 100\n",
    "* batch_size 500"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "```\n",
    "# 케라스를 통해 모델 생성을 시작합니다.\n",
    "\n",
    "model_07 = Sequential()\n",
    "model_07.add(Dense(units=108, input_dim=226, kernel_initializer='he_normal'))\n",
    "model_07.add(Dense(units=108, kernel_initializer='he_normal'))\n",
    "model_07.add(BatchNormalization())\n",
    "model_07.add(Activation('relu'))\n",
    "model_07.add(Dense(units=82, kernel_initializer='he_normal'))\n",
    "model_07.add(Dense(units=82, kernel_initializer='he_normal'))\n",
    "model_07.add(BatchNormalization())\n",
    "model_07.add(Activation('relu'))\n",
    "model_07.add(Dense(units=56, kernel_initializer='he_normal'))\n",
    "model_07.add(Dense(units=56, kernel_initializer='he_normal'))\n",
    "model_07.add(BatchNormalization())\n",
    "model_07.add(Activation('relu'))\n",
    "model_07.add(Dense(units=30, kernel_initializer='he_normal'))\n",
    "model_07.add(Dense(units=30, kernel_initializer='he_normal'))\n",
    "model_07.add(BatchNormalization())\n",
    "model_07.add(Activation('relu'))\n",
    "\n",
    "model_07.add(Dense(units=4, activation='linear'))\n",
    "\n",
    "adam = keras.optimizers.Adam(0.008)\n",
    "model_07.compile(loss='mae', optimizer=adam, metrics=['accuracy'])\n",
    "\n",
    "hist = model_07.fit(train_X, train_Y, epochs=100, batch_size=500, validation_split=0.05)\n",
    "```"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<img src=\"https://i.imgur.com/QqBSvHf.png\" width=\"100%\">"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* BatchNormalization 사이에 layer의 층을 두 배로 늘림으로써 데이터에 최적화시키는 작업과 일반화시키는 작업을 적절히 배치하도록 시도했다.\n",
    "* learning rate를 증가시킴으로써 가중치 업데이트의 보폭을 증가시켰다.\n",
    "* metrics 설정의 문제점을 발견하고 'accuracy'로 설정함으로써 loss와 accuracy를 모두 확인할 수 있도록 했다.\n",
    "* 그래프를 통해서도 알 수 있듯이 validation loss와 accuracy의 변화에 대해서 매우 불안정한 모습을 나타낸다. 또한 Model 6보다 성능이 떨어진다는 것을 확인했다."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### [Model 8](https://github.com/YoonSungLee/DACON-semiconductor-competition-private/blob/master/submission_08.ipynb)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Score : 3.0409400463"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* 11 layers\n",
    "* (239, 252, 265, 178, 91) units, he_normal, relu\n",
    "* BatchNormalization\n",
    "* Adam(0.008)\n",
    "* epochs 100\n",
    "* batch_size 1000"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "```\n",
    "# 케라스를 통해 모델 생성을 시작합니다.\n",
    "\n",
    "model_08 = Sequential()\n",
    "model_08.add(Dense(units= 239, input_dim=226, kernel_initializer='he_normal'))\n",
    "model_08.add(Dense(units= 239, input_dim=226, kernel_initializer='he_normal'))\n",
    "model_08.add(BatchNormalization())\n",
    "model_08.add(Activation('relu'))\n",
    "model_08.add(Dense(units=252, kernel_initializer='he_normal'))\n",
    "model_08.add(Dense(units=252, kernel_initializer='he_normal'))\n",
    "model_08.add(BatchNormalization())\n",
    "model_08.add(Activation('relu'))\n",
    "model_08.add(Dense(units=265, kernel_initializer='he_normal'))\n",
    "model_08.add(Dense(units=265, kernel_initializer='he_normal'))\n",
    "model_08.add(BatchNormalization())\n",
    "model_08.add(Activation('relu'))\n",
    "model_08.add(Dense(units=178, kernel_initializer='he_normal'))\n",
    "model_08.add(Dense(units=178, kernel_initializer='he_normal'))\n",
    "model_08.add(BatchNormalization())\n",
    "model_08.add(Activation('relu'))\n",
    "model_08.add(Dense(units=91, kernel_initializer='he_normal'))\n",
    "model_08.add(Dense(units=91, kernel_initializer='he_normal'))\n",
    "model_08.add(BatchNormalization())\n",
    "model_08.add(Activation('relu'))\n",
    "model_08.add(Dense(units=4, activation='linear'))\n",
    "\n",
    "adam = keras.optimizers.Adam(0.008)\n",
    "model_08.compile(loss='mae', optimizer=adam, metrics=['accuracy'])\n",
    "\n",
    "hist = model_08.fit(train_X, train_Y, epochs=100, batch_size=1000, validation_split=0.05)\n",
    "```"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<img src=\"https://i.imgur.com/QQC4IQb.png\" width=\"100%\">"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* layer의 층을 더 쌓고, 양 쪽 layer units의 평균을 해당 layer units으로 설정했다.\n",
    "* 깊은 layer에 비해 큰 성능을 내지 못했기 때문에 다른 문제점이 있다고 판단했다."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### [Model 9](https://github.com/YoonSungLee/DACON-semiconductor-competition-private/blob/master/submission_09.ipynb)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Score : 8.5367097855"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* 6 layers\n",
    "* (239, 252, 265, 178, 91) units, he_normal, relu\n",
    "* BatchNormalization\n",
    "* Dropout(0.15)\n",
    "* Adam(0.008)\n",
    "* epochs 300\n",
    "* batch_size 1000"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "```\n",
    "# 케라스를 통해 모델 생성을 시작합니다.\n",
    "\n",
    "model_09 = Sequential()\n",
    "model_09.add(Dense(units=239, input_dim=226, kernel_initializer='he_normal'))\n",
    "model_09.add(BatchNormalization())\n",
    "model_09.add(Activation('relu'))\n",
    "model_09.add(Dropout(0.15))\n",
    "model_09.add(Dense(units=252, kernel_initializer='he_normal'))\n",
    "model_09.add(BatchNormalization())\n",
    "model_09.add(Activation('relu'))\n",
    "model_09.add(Dropout(0.15))\n",
    "model_09.add(Dense(units=265, kernel_initializer='he_normal'))\n",
    "model_09.add(BatchNormalization())\n",
    "model_09.add(Activation('relu'))\n",
    "model_09.add(Dropout(0.15))\n",
    "model_09.add(Dense(units=178, kernel_initializer='he_normal'))\n",
    "model_09.add(BatchNormalization())\n",
    "model_09.add(Activation('relu'))\n",
    "model_09.add(Dropout(0.15))\n",
    "model_09.add(Dense(units=91, kernel_initializer='he_normal'))\n",
    "model_09.add(BatchNormalization())\n",
    "model_09.add(Activation('relu'))\n",
    "model_09.add(Dense(units=4, activation='linear'))\n",
    "\n",
    "adam = keras.optimizers.Adam(0.008)\n",
    "model_09.compile(loss='mae', optimizer=adam, metrics=['accuracy'])\n",
    "\n",
    "hist = model_09.fit(train_X, train_Y, epochs=300, batch_size=1000,\n",
    "                    validation_data=(val_X, val_Y))\n",
    "```"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<img src=\"https://i.imgur.com/yHMH1nU.png\" width=\"100%\">"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* Model 8이 깊은 layer에 비해서 그다지 좋은 성능을 내지 못했기 때문에 기존 모델을 다시 복귀시켰다.\n",
    "* Dropout을 다시 도입해서 rate를 줄여 overfitting의 억제를 시도했다.\n",
    "* 대신 epoch를 대폭 늘려 가중치의 업데이트량을 증가시켰다.\n",
    "* sklearn의 train_test_split을 사용하여 train set과 validation set을 랜덤하게 추출했다.\n",
    "* model.fit() 함수의 validation_split 파라미터는 validation set을 train set의 마지막 데이터부터 정해준 비율로 추출한다. 지금까지의 모델은 baseline code를 따라 0.05로 설정했지만, 이 과정에서 문제가 발생했음을 깨달았다. 첫 번째는 validation_split 파라미터의 작동 방식이다. train set은 일정한 규칙에 따라 배열되어 있는 데이터로써 shuffle이 필요하다. 하지만 단순히 파라미터를 통해 validation set을 결정한다면 규칙을 가지고 있는 데이터끼리 결정될것이다. 두 번째는 validation_split의 비율이다. 통상 train set과 validation set은 상황에 따라서 3:1 또는 4:1의 비율로 나눈다. 하지만 baseline code의 비율은 0.05로써 이 비율은 너무나도 작은 값이다. 이는 model이 마치 overfitting이 발생한 것처럼 보이게 할 수 있다. 따라서 sklearn의 train_test_split을 사용했고 그 결과 overfitting이 아니라는 것을 확인할 수 있었다. 즉 지금까지의 모델은 overfitting이 발생하지 않았을 가능성이 크다. 따라서 Dropout의 필요성은 줄어들었고 모델을 더욱 정교하게(복잡하게) 만드는 작업이 필요하다."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### [Model 10](https://github.com/YoonSungLee/DACON-semiconductor-competition-private/blob/master/submission_10.ipynb)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Score : 1.9683300257"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* 11 layers\n",
    "* (239, 252, 265, 178, 91) units, he_normal, swish\n",
    "* BatchNormalization\n",
    "* Adam(0.001)\n",
    "* epochs 200\n",
    "* batch_size 630"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "```\n",
    "# 케라스를 통해 모델 생성을 시작합니다.\n",
    "\n",
    "def create_model():\n",
    "    model = Sequential()\n",
    "    model.add(Dense(units=239, input_dim=226, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=239, kernel_initializer='he_normal'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(Activation(swish))\n",
    "    model.add(Dense(units=252, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=252, kernel_initializer='he_normal'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(Activation(swish))\n",
    "    model.add(Dense(units=265, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=265, kernel_initializer='he_normal'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(Activation(swish))\n",
    "    model.add(Dense(units=178, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=178, kernel_initializer='he_normal'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(Activation(swish))\n",
    "    model.add(Dense(units=91, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=91, kernel_initializer='he_normal'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(Activation(swish))\n",
    "    model.add(Dense(units=4, activation='linear'))\n",
    "\n",
    "    return model\n",
    "```"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "```\n",
    "# Activation Function 정의\n",
    "\n",
    "def swish(x) :\n",
    "    return x * keras.activations.sigmoid(x)\n",
    "```"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "```\n",
    "# BayesianOptimization 객체 생성, 실행 및 최종 결과 출력\n",
    "# 특정 모델에서 최적의 learning_rate와 batch_size를 찾기 위함\n",
    "# model.fit() 의 verbose를 0으로 설정하여 \"Buffered data was truncated after reaching the output size limit.\" 문제 해결\n",
    "\n",
    "bayes_optimizer = BayesianOptimization(\n",
    "    f=train_and_validate,\n",
    "    pbounds={\n",
    "        'learning_rate' : (0.001, 0.1),\n",
    "        'batch_size' : (500,1000)\n",
    "    },\n",
    "    random_state=0,\n",
    "    verbose=2\n",
    ")\n",
    "\n",
    "bayes_optimizer.maximize(init_points=3, n_iter=27, acq='ei',xi=0.01)\n",
    "\n",
    "for i, res in enumerate(bayes_optimizer.res):\n",
    "    print('iteration {}: \\n\\t{}'.format(i, res))\n",
    "print('Final result: ', bayes_optimizer.max)\n",
    "```"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<img src=\"https://i.imgur.com/XzWC3ki.png\" width=\"100%\">"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* 여러 개의 모델을 생성하는 과정을 반복하다보면 이 과정이 불편함을 느낀다. 이러한 경험을 통해 모델 생성을 함수화시키는 것이 필요하다는 것을 느껴서 함수로 구현했다. 만약 모델의 구성을 완벽히 했다면, 주요 파라미터들을 인풋으로 받는 함수를 정의한다면, 손쉽게 다양한 모델을 생성할 수 있을 것이다.\n",
    "* 서론에서 언급했듯이 수많은 Machine Learning Model, 그리고 Deep Learning Model은 하이퍼파라미터(learning rate, batchsize, L2 정규화계수, units, layer의 수, epoch 등) 설정이 중요한 이슈이다. 모델의 하이퍼파라미터 결정 방법은 크게 4가지로 구분된다. 첫 번째 방법은 'Manual Search'이다. 이는 설계자의 직관에 의존하는 방법으로써 대중적으로 알려진 노하우, 또는 노력에 의해 하이퍼파라미터를 결정한다. 심리상 이러한 과정을 통해 얻은 결과가 최선의 결과임을 부정하기는 매우 어려운 일이다. 따라서 많은 공을 들여 하이퍼파라미터를 결정하면 이후에는 쉽게 그 값을 바꾸려 하지 않는다는 단점이 있다. 또한 둘 이상의 하이퍼파라미터를 탐색한다면 상호영향 문제때문에 단순히 사람의 조작으로 쉽게 최적화된 하이퍼파라미터를 찾기가 어렵다. 두 번째 방법은 'Grid Search'이다. 이는 특정 구간 내의 후보 하이퍼파라미터들을 일정 간격을 두고 선정하는 방법으로 균등하고 전역적이라는 특징을 가지고 있다. Manual Search보다는 체계적이지만 여러 종류의 하이퍼파라미터를 결정하는 탐색을 한다면 시간이 너무 오래 걸린다는 단점이 있다. 세 번째 방법은 'Random Search'이다. 이는 특정 구간 내의 후보 하이퍼파라미터들을 랜덤샘플링을 통해 결정하는 방법이다. Grid Search에 비해 랜덤성을 부여함으로써 최적화된 하이퍼파라미터들을 더 잘 찾을 수 '도' 있다. 하지만 이 또한 운이 따라야 하는 방법임은 부정할 수 없다. 좋은 하이퍼파라미터를 찾기 위해서는 기존의 시도들을 분석하여 올바른 방법으로 다음 시도를 결정해야 한다. 즉, 사전지식이 사후결정에 반영되어야 한다. 이 방법이 네 번째 방법으로 'Bayesian Optimization'이다. 이번 조사를 통해 베이즈 정리, 나이브 베이즈 알고리즘 등 '베이즈'라는 단어는 '사전 지식'의 의미를 담고 있음을 나의 사전 지식을 통해 확인할 수 있었다. 자, 그래서 이 방법은 목적함수 f를 설정하여 기존의 탐색한 기록들을 바탕으로 f(x)를 계속해서 추측해내고, 추측해낸 f(x)를 최대로 만드는 최적해 x를 다음 탐색점으로 결정한다. 특히 이 방법은 '착취'와 '탐색' 방법을 이용한다. 착취는 지금까지의 최댓값 x 주변에 더 큰 최댓값이 있다고 예상하여 그 주변의 값들을 후보로 사용하는 방법이고, 탐색은 지금까지 확인하지 못했던 미지의 구간에 오히려더 큰 최댓값이 있다고 예상하여 그 값들을 후보로 사용하는 방법이다. 이번 모델에서는 네 번째 방법인 Bayesian Optimization을 사용하여 최적의 batch_size와 learning_rate를 찾아보기로 했다.\n",
    "* Bayesian Optimization 탐색 결과 최적의 batch_size는 630, learning_rate는 0.001임을 알 수 있었다.\n",
    "* activation function을 relu에서 swish로 변경했다. train data를 살펴보면 음수 데이터들이 존재함을 확인할 수 있는데, relu는 이런 음수값들을 0으로 만들어준다는 단점이 있다. 따라서 0과 가까운 음수값들은 작은 가중치를 두어 그 값을 살리는 swish함수가 activation function으로 적합하다고 판단했다."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Bayesian Optimization과 Swish에 관한 내용은 아래 사이트에서 더 구체적인 정보를 얻을 수 있다."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Bayesian Optimization\n",
    "http://research.sualab.com/introduction/practice/2019/02/19/bayesian-optimization-overview-1.html\n",
    "http://research.sualab.com/introduction/practice/2019/04/01/bayesian-optimization-overview-2.html"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Swish Activation\n",
    "https://www.machinecurve.com/index.php/2019/05/30/why-swish-could-perform-better-than-relu/#todays-activation-functions"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### [Model 11](https://github.com/YoonSungLee/DACON-semiconductor-competition-private/blob/master/submission_11.ipynb)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Score : 2.3582599163"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* 13 layers\n",
    "* (201, 176, 151, 126, 101, 76) units, he_normal, swish\n",
    "* BatchNormalization\n",
    "* Adam(0.001)\n",
    "* epochs 200\n",
    "* batch_size 630"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "```\n",
    "# 케라스를 통해 모델 생성을 시작합니다.\n",
    "\n",
    "def create_model():\n",
    "    model = Sequential()\n",
    "    model.add(Dense(units=201, input_dim=226, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=201, kernel_initializer='he_normal'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(Activation(swish))\n",
    "    model.add(Dense(units=176, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=176, kernel_initializer='he_normal'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(Activation(swish))\n",
    "    model.add(Dense(units=151, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=151, kernel_initializer='he_normal'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(Activation(swish))\n",
    "    model.add(Dense(units=126, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=126, kernel_initializer='he_normal'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(Activation(swish))\n",
    "    model.add(Dense(units=101, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=101, kernel_initializer='he_normal'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(Activation(swish))\n",
    "    model.add(Dense(units=76, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=76, kernel_initializer='he_normal'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(Activation(swish))\n",
    "    model.add(Dense(units=4, activation='linear'))\n",
    "\n",
    "    return model\n",
    "```"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<img src=\"https://i.imgur.com/t7PUNlg.png\" width=\"100%\">"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* Bayesian Optimization에서 도출한 learning_rate와 batch_size는 그대로 사용한다.\n",
    "* 총 13개의 layer를 쌓고 2개 단위로 양쪽 layer units의 평균을 해당 layer units으로 설정했다.\n",
    "* layer를 더 깊게 쌓았음에도 불구하고 Model 10보다 좋지 않은 성능을 냈음을 확인했다. 이를 통해 units의 수를 결정하는 방법에 문제가 있음을 확인했고 적절한 units 수를 결정하는 것이 중요하다고 생각했다."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### [Find Optimal Units](https://github.com/YoonSungLee/DACON-semiconductor-competition-private/blob/master/find_optimal_units.ipynb)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "```\n",
    "# 케라스를 통해 모델 생성을 시작합니다.\n",
    "\n",
    "def create_model(units):\n",
    "    model = Sequential()\n",
    "    model.add(Dense(units=units, input_dim=226, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=units, kernel_initializer='he_normal'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(Activation(swish))\n",
    "    model.add(Dense(units=units, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=units, kernel_initializer='he_normal'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(Activation(swish))\n",
    "    model.add(Dense(units=units, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=units, kernel_initializer='he_normal'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(Activation(swish))\n",
    "    model.add(Dense(units=units, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=units, kernel_initializer='he_normal'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(Activation(swish))\n",
    "    model.add(Dense(units=units, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=units, kernel_initializer='he_normal'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(Activation(swish))\n",
    "    model.add(Dense(units=4, activation='linear'))\n",
    "\n",
    "    return model\n",
    "\n",
    "\n",
    "# Activation Function 정의\n",
    "\n",
    "def swish(x) :\n",
    "    return x * keras.activations.sigmoid(x)\n",
    "```"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "```\n",
    "# layer의 특정 units 하에서 학습을 수행한 후, 검증 성능을 출력하는 목적 함수 정의\n",
    "\n",
    "def train_and_validate(units):\n",
    "    model = create_model(int(units))\n",
    "    adam = keras.optimizers.Adam(0.001)\n",
    "    model.compile(loss='mae', optimizer=adam, metrics=['accuracy'])\n",
    "    hist = model.fit(train_X, train_Y, epochs=20, batch_size=630,\n",
    "                    validation_data=(val_X, val_Y))\n",
    "    best_val_score = max(hist.history['val_acc'])\n",
    "\n",
    "    return best_val_score\n",
    "```"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "```\n",
    "# BayesianOptimization 객체 생성, 실행 및 최종 결과 출력\n",
    "# 최적의 layer units 수를 찾기 위함\n",
    "\n",
    "bayes_optimizer = BayesianOptimization(\n",
    "    f=train_and_validate,\n",
    "    pbounds={\n",
    "         'units' : (100, 800)\n",
    "    },\n",
    "    random_state=0,\n",
    "    verbose=1\n",
    ")\n",
    "\n",
    "bayes_optimizer.maximize(init_points=3, n_iter=27, acq='ei',xi=0.01)\n",
    "\n",
    "for i, res in enumerate(bayes_optimizer.res):\n",
    "    print('iteration {}: \\n\\t{}'.format(i, res))\n",
    "```"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "```\n",
    "print('Final result: ', bayes_optimizer.max)\n",
    "```"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Final result:  {'target': 0.9581975173950196, 'params': {'units': 799.999308421973}}"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Model11을 통해 지금까지 설정했던 units이 문제가 있음을 직감적으로 알 수 있었고, units의 수 또한 learning_rate와 batch_size를 결정했던 방법처럼 BayesianOptimization()을 이용하여 최적의 값을 탐색하기로 했다. 그 결과 적절한 units수는 800임을 확인했고, 이는 AI Study를 통해 다른 팀원이 Manual Search에 의하여 도출한 값과 동일했다. 따라서 더욱 확신을 갖고 units의 수를 이 값으로 고정시키기로 했다."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### [Model 12](https://github.com/YoonSungLee/DACON-semiconductor-competition-private/blob/master/submission_12.ipynb)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Score : 1.2427999973"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* 11 layers\n",
    "* 800 units, he_normal, swish\n",
    "* BatchNormalization\n",
    "* Adam(0.001)\n",
    "* epochs 200\n",
    "* batch_size 630"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "```\n",
    "\n",
    "# 케라스를 통해 모델 생성을 시작합니다.\n",
    "\n",
    "def create_model():\n",
    "    model = Sequential()\n",
    "    model.add(Dense(units=800, input_dim=226, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(Activation(swish))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(Activation(swish))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(Activation(swish))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(Activation(swish))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(Activation(swish))\n",
    "    model.add(Dense(units=4, activation='linear'))\n",
    "\n",
    "    return model\n",
    "```"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<img src=\"https://i.imgur.com/y1qjav3.png\" width=\"100%\">"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* BayesianOptimization을 통해 도출한 최적의 units을 적용했다.\n",
    "* 앞선 모델들보다 확실히 좋은 점수를 받았고, 이후 할 일은 현재 모델에서 성능을 더욱 좋게 나오게 할 방법을 찾는 일이다."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### [Model 13](https://github.com/YoonSungLee/DACON-semiconductor-competition-private/blob/master/submission_13.ipynb)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Score : 1.2043600082"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* 13 layers\n",
    "* 800 units, he_normal, swish\n",
    "* BatchNormalization\n",
    "* Adam(0.001)\n",
    "* epochs 200\n",
    "* batch_size 630"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "```\n",
    "def create_model():\n",
    "    model = Sequential()\n",
    "    model.add(Dense(units=800, input_dim=226, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(Activation(swish))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(Activation(swish))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(Activation(swish))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(Activation(swish))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(Activation(swish))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(Activation(swish))\n",
    "    model.add(Dense(units=4, activation='linear'))\n",
    "\n",
    "    return model\n",
    "```"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<img src=\"https://i.imgur.com/7CCWjaj.png\" width=\"100%\">"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* layer의 층을 더 깊게하여 모델을 구성했다."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### [Model 14](https://github.com/YoonSungLee/DACON-semiconductor-competition-private/blob/master/submission_14.ipynb)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Score : 1.0792399645"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* 15 layers\n",
    "* 800 units, he_normal, swish\n",
    "* BatchNormalization\n",
    "* Adam(0.001)\n",
    "* epochs 200\n",
    "* batch_size 630"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "```\n",
    "# 케라스를 통해 모델 생성을 시작합니다.\n",
    "\n",
    "def create_model():\n",
    "    model = Sequential()\n",
    "    model.add(Dense(units=800, input_dim=226, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(Activation(swish))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(Activation(swish))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(Activation(swish))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(Activation(swish))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(Activation(swish))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(Activation(swish))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(Activation(swish))\n",
    "    model.add(Dense(units=4, activation='linear'))\n",
    "\n",
    "    return model\n",
    "```"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<img src=\"https://i.imgur.com/xBnO9rv.png\" width=\"100%\">"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* layer의 층을 더 깊게하여 모델을 구성했다."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### [Model 15](https://github.com/YoonSungLee/DACON-semiconductor-competition-private/blob/master/submission_15.ipynb)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Score : 1.6183999777"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* 19 layers\n",
    "* 800 units, he_normal, swish\n",
    "* BatchNormalization\n",
    "* Adam(0.001)\n",
    "* epochs 200\n",
    "* batch_size 630"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "```\n",
    "# 케라스를 통해 모델 생성을 시작합니다.\n",
    "\n",
    "def create_model():\n",
    "    model = Sequential()\n",
    "    model.add(Dense(units=800, input_dim=226, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(Activation(swish))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(Activation(swish))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(Activation(swish))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(Activation(swish))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(Activation(swish))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(Activation(swish))\n",
    "    model.add(Dense(units=4, activation='linear'))\n",
    "\n",
    "    return model\n",
    "```"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<img src=\"https://i.imgur.com/hC4RiOy.png\" width=\"100%\">"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* BatchNormalization 사이의 층을 깊게 모델링하는 방법을 시도했다.\n",
    "* Score를 통해 알 수 있듯이 오히려 성능이 떨어짐을 확인할 수 있다. [layer]-[layer]-[BatchNormalization]이 적절한 모델링 구성 방법임을 확인했다."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### [Model 16](https://github.com/YoonSungLee/DACON-semiconductor-competition-private/blob/master/submission_16.ipynb)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Score : 0.6688299775"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "```\n",
    "# 좋은 성능을 보였던 모델을 모두 불러옵니다.\n",
    "\n",
    "from keras.models import load_model\n",
    "\n",
    "model_12_200 = load_model('/gdrive/My Drive/DACON-semiconductor-competition/model_12.h5', custom_objects={'swish':swish})\n",
    "model_12_300 = load_model('/gdrive/My Drive/DACON-semiconductor-competition/model_12_300.h5', custom_objects={'swish':swish})\n",
    "model_13_200 = load_model('/gdrive/My Drive/DACON-semiconductor-competition/model_13.h5', custom_objects={'swish':swish})\n",
    "model_13_250 = load_model('/gdrive/My Drive/DACON-semiconductor-competition/model_13_250.h5', custom_objects={'swish':swish})\n",
    "model_13_300 = load_model('/gdrive/My Drive/DACON-semiconductor-competition/model_13_300.h5', custom_objects={'swish':swish})\n",
    "model_14_200 = load_model('/gdrive/My Drive/DACON-semiconductor-competition/model_14.h5', custom_objects={'swish':swish})\n",
    "```"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "```\n",
    "# 예측값을 생성합니다.\n",
    "\n",
    "model_predict = []\n",
    "for model in model_list:\n",
    "    model_predict.append(model.predict(test_X))\n",
    "\n",
    "pred_test = sum(model_predict)/6\n",
    "```"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* 지금까지 좋은 성능을 보였던 모델들을 모두 불러와 (각 모델들의 예측값들의 합)/(모델의 수) 를 계산하여 '앙상블 기법'을 적용한 결과값 도출했다.\n",
    "* 굉장히 좋은 Score를 얻을 수 있었고, 이를 통해 앙상블 기법의 효과를 가늠할 수 있었다. 성능이 아주 좋은 모델 하나보다, 적절한 성능을 가진 모델 여러개를 종합해서 나온 결과가 더 좋은 결과를 도출한다."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### [Model 17](https://github.com/YoonSungLee/DACON-semiconductor-competition-private/blob/master/submission_17.ipynb)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Score : 1.1050399542"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* 17 layers\n",
    "* 800 units, he_normal, swish\n",
    "* BatchNormalization\n",
    "* Adam(0.001)\n",
    "* epochs 200\n",
    "* batch_size 630"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "```\n",
    "# 케라스를 통해 모델 생성을 시작합니다.\n",
    "\n",
    "def create_model():\n",
    "    model = Sequential()\n",
    "    model.add(Dense(units=800, input_dim=226, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(Activation(swish))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(Activation(swish))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(Activation(swish))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(Activation(swish))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(Activation(swish))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(Activation(swish))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(Activation(swish))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(Activation(swish))\n",
    "    model.add(Dense(units=4, activation='linear'))\n",
    "\n",
    "    return model\n",
    "```"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<img src=\"https://i.imgur.com/avyZrgy.png\" width=\"100%\">"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* 앙상블 모델에 추가로 적용하기 위해 layer를 더더욱 깊게 모델링했다."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### [Model 18](https://github.com/YoonSungLee/DACON-semiconductor-competition-private/blob/master/submission_18.ipynb)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* 17 layers\n",
    "* 800 units, he_normal, swish\n",
    "* BatchNormalization\n",
    "* Adam(0.001)\n",
    "* epochs 200\n",
    "* batch_size 630"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "```\n",
    "# 케라스를 통해 모델 생성을 시작합니다.\n",
    "\n",
    "def create_model():\n",
    "    model = Sequential()\n",
    "    model.add(Dense(units=800, input_dim=226, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(Activation(swish))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(Activation(swish))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(Activation(swish))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(Activation(swish))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(Activation(swish))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(Activation(swish))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(BatchNormalization())\n",
    "    model.add(Activation(swish))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(Dense(units=800, kernel_initializer='he_normal'))\n",
    "    model.add(Activation(swish))\n",
    "    model.add(Dense(units=4, activation='linear'))\n",
    "\n",
    "    return model\n",
    "```"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "<img src=\"https://i.imgur.com/jxqn1uU.png\" width=\"100%\">"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* 기존과 같은 방법으로 layer를 깊게 모델링했고, 마지막 층에서만 BatchNormalization을 제외시키는 방법을 이용했다.\n",
    "* 별다른 성과는 얻지 못했다."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### [Model 19](https://github.com/YoonSungLee/DACON-semiconductor-competition-private/blob/master/submission_19.ipynb)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Score : 0.6301199794"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "```\n",
    "# 좋은 성능을 보였던 모델을 모두 불러옵니다.\n",
    "\n",
    "from keras.models import load_model\n",
    "\n",
    "model_12_200 = load_model('/gdrive/My Drive/DACON-semiconductor-competition/model_12.h5', custom_objects={'swish':swish})\n",
    "model_12_300 = load_model('/gdrive/My Drive/DACON-semiconductor-competition/model_12_300.h5', custom_objects={'swish':swish})\n",
    "model_13_200 = load_model('/gdrive/My Drive/DACON-semiconductor-competition/model_13.h5', custom_objects={'swish':swish})\n",
    "model_13_250 = load_model('/gdrive/My Drive/DACON-semiconductor-competition/model_13_250.h5', custom_objects={'swish':swish})\n",
    "model_13_300 = load_model('/gdrive/My Drive/DACON-semiconductor-competition/model_13_300.h5', custom_objects={'swish':swish})\n",
    "model_14_200 = load_model('/gdrive/My Drive/DACON-semiconductor-competition/model_14.h5', custom_objects={'swish':swish})\n",
    "model_15_200 = load_model('/gdrive/My Drive/DACON-semiconductor-competition/model_15.h5', custom_objects={'swish':swish})\n",
    "model_17_200 = load_model('/gdrive/My Drive/DACON-semiconductor-competition/model_17.h5', custom_objects={'swish':swish})\n",
    "model_18_200 = load_model('/gdrive/My Drive/DACON-semiconductor-competition/model_18.h5', custom_objects={'swish':swish})\n",
    "```"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "```\n",
    "# 예측값을 생성합니다.\n",
    "\n",
    "model_predict = []\n",
    "for model in model_list:\n",
    "    model_predict.append(model.predict(test_X))\n",
    "\n",
    "pred_test = sum(model_predict)/len(model_predict)\n",
    "```"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "* 지금까지 좋은 성능을 보였던 모델들을 모두 불러와 (각 모델들의 예측값들의 합)/(모델의 수) 를 계산하여 '앙상블 기법'을 적용한 결과값을 도출했다.\n",
    "* 최종적으로 이 결과값을 이용한 Score를 제출했다."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 4. 결과"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "![image.png](https://i.imgur.com/GO03sdT.png)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "최종 결과 제출횟수 18회, 점수 0.6181722715점, 등수 26등을 기록했다. 이 정보를 확인할 수 있는 사이트는 아래와 같다."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "https://dacon.io/competitions/official/235554/leaderboard/"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 5. 고찰"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "AI Study를 모집하여 개념에 대한 공부를 하다가 이번에 처음으로 Deep Learning 분야에 대한 Competition에 참가했다. 처음인만큼 모델링을 하면서 실수도 많았고 체계적이지 못한 접근법도 많았다. 하지만 또 반대로 지금까지 공부했던 내용들을 실제로 적용해보면서 이론상으로만 알고 있던 지식들을 몸소 체감하는 기회를 가질 수 있었다. 이번 Competition을 준비하면서 경험했던 특별한 개념들, 실수들을 정립하고 스스로를 피드백함으로써 역량을 키우고자 한다."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### MLP에 의존한 모델링\n",
    "<br>\n",
    "처음 모델링을 시작한만큼 딥러닝 프레임워크 사용법에 대한 공부가 필요했기에 Keras를 선택해 속성으로 공부했다. 여러 모델을 적용하기에 Keras에 대한 숙지가 충분하지 않아서 결국 MLP 모델의 성능을 올리는 데 주력했다. 하지만 이번 Competition은 데이터 사이에 상관관계가 높은 특성을 가지고 있기에 MLP보다는 CNN이나 RNN(또는 LSTM)이 적합하다고 생각한다. 특히 스터디원 중 CNN 모델을 구성한 인원은 상위 등수를 받을 수 있었다. Keras 사용법을 숙지해서 MLP, CNN, RNN 등의 모델에 대하여 기초 모델링을 준비해놓을 필요가 있다. 또한 이번 데이터는 Deep Learning에 적합한 데이터이지만 Machine Learning 모델을 적용해보는 것도 하나의 방안이 될 수 있을 것이다."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### summary, callback 등 주요한 Keras 도구 숙지\n",
    "<br>\n",
    "모델의 아키텍처를 확인하는 방법으로 SVG와 model_to_dot을 사용했는데 적절치 못하다는 판단을 내렸다. 이미지의 사이즈가 너무 커서 모델의 전체적인 모습을 확인하지 못했고, 그 외의 정보들 또한 얻기가 어려웠다. Keras에서는 model.summary의 간단한 코드로 모델의 아키텍처, weight의 수 등을 보기 쉽게 정리해준다. 또한 callback함수를 뒤늦게 알게 되었는데, 이를 통해 overfitting이 발생하기 전까지 학습한 모델과 그 weight를 저장할 수 있다. 이렇게 Keras는 사용자가 원하는 것들을 간편하게 구현할 수 있도록 해놓았다. 이런 부분에 대한 숙지가 필요하다."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### 체계적이지 않은 모델링과 하이퍼파라미터 탐색 방법\n",
    "<br>\n",
    "위의 내용을 보면 알 수 있듯이 굉장히 많은 모델링을 했다. 하지만 자세히 보면 체계적이지 않은 모델링과 하이퍼파라미터 탐색 방법 때문이다. 모델링을 함수를 통해 정의하는 것, 앞으로 모델링해야할 목록들을 구상해놓고 하나씩 소거하는 것, Grid Search, Random Search, BayesianOptimization Search 등 하이퍼파라미터 탐색 방법을 숙지할 필요가 있다. 특히 모델마다 적합한 하이퍼파라미터가 다르기 때문에 모델이 바뀌면 다시 하이퍼파라미터를 탐색해야 한다는 것을 인지할 필요가 있다."
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
