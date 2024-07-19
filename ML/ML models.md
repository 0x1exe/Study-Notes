# SVM
### Линейная  SVM
Главная цель SVM как классификатора — найти уравнение разделяющей гиперплоскости  
![$w_1x_1+w_2x_2+…+w_nx_n+w_0=0$](https://habrastorage.org/getpro/habr/formulas/440/bf4/453/440bf445316d01f98a578314a16e6064.svg) в пространстве ![$R^n$](https://habrastorage.org/getpro/habr/formulas/3a4/f52/2b1/3a4f522b143bbf174a2d3805dff7536d.svg), которая бы разделила два класса неким оптимальным образом. Общий вид преобразования ![$F$](https://habrastorage.org/getpro/habr/formulas/a0d/d16/4e4/a0dd164e481befe52ca1b226f287b94e.svg) объекта ![$x$](https://habrastorage.org/getpro/habr/formulas/4cc/fd4/32e/4ccfd432ea4f2a64f3a5c8c7378517af.svg) в метку класса ![$Y$](https://habrastorage.org/getpro/habr/formulas/53a/ea9/f07/53aea9f07ccaf30ffac7cd8719e70972.svg): ![$F(x) = sign(w^Tx-b)$](https://habrastorage.org/getpro/habr/formulas/f68/424/abf/f68424abf4aaf9d977999f77eeb3629b.svg). Будем помнить, что мы обозначили ![$w = (w_1, w_2, …, w_n), b=-w_0$](https://habrastorage.org/getpro/habr/formulas/7a9/779/9fa/7a97799fa14451b0d95466eb65cc35df.svg). После настройки весов алгоритма ![$w$](https://habrastorage.org/getpro/habr/formulas/499/78e/f12/49978ef12ee6820ac7fc4607771a3586.svg) и ![$b$](https://habrastorage.org/getpro/habr/formulas/39d/180/62d/39d18062d6d75592f56b1b38409a5e10.svg) (обучения), все объекты, попадающие по одну сторону от построенной гиперплоскости, будут предсказываться как первый класс, а объекты, попадающие по другую сторону — второй класс.


Дефолтную настройку _SVM с жестким зазором_ (_hard-margin SVM_), когда никакому объекту не разрешается попадать на полосу разделения можно выразить следующим образом. Решается аналитически через теорему Куна-Таккера. Получаемая задача эквивалентна двойственной задаче поиска седловой точки функции Лагранжа.

![$ \left\{ \begin{array}{ll} (w^Tw)/2 \rightarrow min & \textrm{}\\ y(w^Tx-b) \geqslant 1 & \textrm{} \end{array} \right. $](https://habrastorage.org/getpro/habr/formulas/cf8/a48/720/cf8a487200e1dafca82cba771de420ed.svg)

### Non-linear SVM
Позволим алгоритму допускать ошибки на обучающих объектах, но при этом постараемся, чтобы ошибок было поменьше. Введём набор дополнительных переменных ![$\xi _i > 0$](https://habrastorage.org/getpro/habr/formulas/df8/92a/2dd/df892a2ddd95ef764b9059625f9594ba.svg), характеризующих величину ошибки на каждом объекте ![$x_i$](https://habrastorage.org/getpro/habr/formulas/341/585/9a0/3415859a0c4e2dbfd25b06a38e760de3.svg). Введём в минимизируемый функционал штраф за суммарную ошибку:


![$ \left\{ \begin{array}{ll} (w^Tw)/2 + \alpha\sum\xi _i \rightarrow min & \textrm{}\\ y(w^Tx_i-b) \geqslant 1 -\xi _i & \textrm{}\\ \xi _i\geqslant0& \textrm{} \end{array} \right. $](https://habrastorage.org/getpro/habr/formulas/71a/823/9c4/71a8239c468609b7a59aea01d6c26e8d.svg)

Будем считать количество ошибок алгоритма (когда M<0). Назовем это штрафом (_Penalty_).
При добавлении к выражению штрафа слагаемое ![$\alpha(w^Tw)/2$](https://habrastorage.org/getpro/habr/formulas/d64/3fd/4a5/d643fd4a5839b0861f29cbc93ea2d0be.svg) получаем классическую фукцию потерь _SVM с мягким зазором_ (_soft-margin SVM_) для одного объекта:

![$Q =max(0,1- M_i) + \alpha(w^Tw)/2$](https://habrastorage.org/getpro/habr/formulas/5ee/54f/23b/5ee54f23b5ef34510619d757a75f1a66.svg)
![$Q =max(0,1- yw^Tx) + \alpha(w^Tw)/2$](https://habrastorage.org/getpro/habr/formulas/9ee/a84/312/9eea843122a687869f548015a1067002.svg)
![$Q$](https://habrastorage.org/getpro/habr/formulas/3a5/e67/187/3a5e67187c94ffba1bacdbc00b809d08.svg) — функция потерь, она же loss function. Именно ее мы и будем минимизировать с помощью градиентного спуска в реализации руками.

### Kernels

Nonlinear SVM addresses linear limitation by utilizing kernel functions to map the data into a higher-dimensional space where linear separation becomes possible.

The kernel function computes the similarity between data points, allowing SVM to capture complex patterns and nonlinear relationships between features. This enables nonlinear SVM to handle intricate data distributions, such as curved or circular decision boundaries.

![[Pasted image 20240218170343.png]]


Once the dual problem is solved and the optimal Lagrange multipliers are determined, the SVM decision boundary can be expressed in terms of these optimal Lagrange multipliers and the support vectors. The support vectors are the training samples with i > 0, and the decision boundary is given by:

![[Pasted image 20240218170515.png]]

The SVM kernel is a function that transforms low-dimensional input space into higher-dimensional space, or in other words, it turns nonseparable problems into separable problems:

![[Pasted image 20240218170542.png]]


# Guide to Ensembling methods

**A group of predictors is called an ensemble**; thus, this technique is called Ensemble Learning, and an Ensemble Learning algorithm is called an **Ensemble method**

## **Types of ensembling :**

**Basic Ensemble Techniques**

- Max Voting
- Averaging
- Weighted Average

**Advanced Ensemble Techniques**

- Stacking
- Blending
- Bagging
- Boosting
**Algorithms based on Bagging and Boosting**

> - Bagging meta-estimator
> - Random Forest
> - AdaBoost
> - GBM
> - XGB
> - Light GBM
> - CatBoost



## Errors
The error emerging from any model can be broken into three components:

![[Pasted image 20240405154852.png]]

**Bias error**

is useful to quantify how much on an average are the predicted values different from the actual value. A high bias error means we have a under-performing model which keeps on missing important trends.

**Variance**

on the other side quantifies how are the prediction made on same observation different from each other. A high variance model will over-fit on your training population and perform badly on any observation beyond training. Following diagram will give you more clarity (Assume that red spot is the real value and blue dots are predictions) :

![[Pasted image 20240405155230.png]]![[Pasted image 20240405155255.png]]

## Max voting

**Implementation**:
```
model1 = tree.DecisionTreeClassifier()
model2 = KNeighborsClassifier()
model3= LogisticRegression()

model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)

pred1=model1.predict(x_test)
pred2=model2.predict(x_test)
pred3=model3.predict(x_test)

final_pred = np.array([])
for i in range(0,len(x_test)):
    final_pred = np.append(final_pred, mode([pred1[i], pred2[i], pred3[i]]))
```
OR (built-in):
```
from sklearn.ensemble import VotingClassifier
model1 = LogisticRegression(random_state=1)
model2 = tree.DecisionTreeClassifier(random_state=1)
model = VotingClassifier(estimators=[('lr', model1), ('dt', model2)], voting='hard')
model.fit(x_train,y_train)
model.score(x_test,y_test)
```
## Majority voting/Hard voting

**Soft Voting**

IF all classifiers are able to estimate class probabilities (i.e. they have a predict_proba() method), then you can tell Scikit-Learn to predict the class with the highest class probability, averaged over all the individual classifiers.

This is called **soft voting** and it often achieves higher performance than hard voting because _it gives more weight to highly confident votes_.

**In soft voting**, we predict the class labels based on the predicted probabilities p for classifier -- this approach is only recommended if the classifiers are **well-calibrated**.

_y^=argmax∑j=1mwjpij,_ where **wj** is the weight that can be assigned to the **jth** classifier.

Assuming the example in the previous section was a _binary classification_ task with class labels i∈{0,1}, our ensemble could make the following prediction:

C1(x)→[0.9,0.1]

C2(x)→[0.8,0.2]

C3(x)→[0.4,0.6]

Using uniform weights, we compute the average probabilities:

p(i0∣x)=0.9+0.8+0.43=0.7p(i1∣x)=0.1+0.2+0.63=0.3

y^=argmax[p(i0∣x),p(i1∣x)]=0


**Averaging**

Similar to the max voting technique, multiple predictions are made for each data point in averaging. In this method, we take an **average** of predictions from all the models and use it to make the final prediction. Averaging can be used for making predictions in regression problems or while calculating probabilities for classification problems. For example, in the below case, the averaging method would take the average of all the values.

```
model1 = tree.DecisionTreeClassifier()
model2 = KNeighborsClassifier()
model3= LogisticRegression()

model1.fit(x_train,y_train)
model2.fit(x_train,y_train)
model3.fit(x_train,y_train)

pred1=model1.predict_proba(x_test)
pred2=model2.predict_proba(x_test)
pred3=model3.predict_proba(x_test)

finalpred=(pred1+pred2+pred3)/3
```


**Weighted averaging**

```
finalpred=(pred1*0.3+pred2*0.3+pred3*0.4)
```

## Advanced techniques

### Bagging

In order for this to work, your data must have _variance_, otherwise you’re just adding levels after levels of additional iterations with **little benefit** to your score and a big headache for those maintaining your modeling pipeline in production. Even when it does improve things, you have to ask yourself if its worth all that extra work…

In simple terms, **bagging irons out variance from a data set** . If, after splitting your data into multiple chunks and training them, you find that your predictions are _different_, then your data has _variance_. Bagging can turn a bad thing into a competitive advantage. For more theory behind the magic, check out _Bootstrap Aggregating on Wikipedia._ Bagging was invented by _Leo Breiman_ at the University of California. He is also one of the grandfathers of Boosting and Random Forests.

**Stability and Accuracy**

By saving each prediction set and averaging them together, you not only lower variance without affecting bias, but your accuracy may be **improved**! In essence, you are creating many slightly different models and ensembling them together; **this avoids over-fitting**, **stabilizes your predictions and increases your accuracy**. Mind you, this assumes your data has variance, if it doesn’t,**bagging won’t help.**

**Bagging algorithms:**

- Bagging meta-estimator
- Random forest

**Bagging meta-estimator** is an ensembling algorithm that can be used for **both** classification (BaggingClassifier) and regression (BaggingRegressor) problems. It follows the typical bagging technique to make predictions. Following are the steps for the bagging meta-estimator algorithm:

1-Random subsets are created from the original dataset (Bootstrapping).

2-The subset of the dataset includes all features.

3-A user-specified base estimator is fitted on each of these smaller sets.

4-Predictions from each model are combined to get the final result.

```
final_dt = DecisionTreeClassifier(max_leaf_nodes=10, max_depth=5)                   
final_bc = BaggingClassifier(base_estimator=final_dt, n_estimators=40, random_state=1, oob_score=True)

final_bc.fit(X_train, train_y)
final_preds = final_bc.predict(X_test)

acc_oob = final_bc.oob_score_
print(acc_oob)
```

**Random Forest** is another ensemble machine learning algorithm that follows the bagging technique. It is an extension of the bagging estimator algorithm. The base estimators in random forest are decision trees. Unlike bagging meta estimator, random forest **randomly** selects a set of features which are used to decide the best split at each node of the decision tree.

step-by-step, this is what a random forest model does:

1-Random subsets are created from the original dataset (bootstrapping).

2-At each node in the decision tree, only a random set of features are considered to decide the best split.

3-A decision tree model is fitted on each of the subsets. The final prediction is calculated by averaging the predictions from all decision trees.

**Note:** The decision trees in random forest can be built on a subset of data and features. Particularly, the sklearn model of random forest uses all features for decision tree and a subset of features are randomly selected for splitting at each node.
### Boosting

The term ‘Boosting’ refers to a family of algorithms which **converts weak learner to strong learners**. Boosting is an ensemble method for improving the model predictions of any given learning algorithm. The idea of boosting **is to train weak learners sequentially, each trying to correct its predecessor**.

**Adaptive boosting or AdaBoost** is one of the simplest boosting algorithms. Usually, decision trees are used for modelling. Multiple sequential models are created, each correcting the errors from the last model. AdaBoost assigns weights to the observations which are incorrectly predicted and the subsequent model works to predict these values correctly.

**steps:**

1-all observations in the dataset are given equal weights.

2-A model is built on a subset of data.

3-Using this model, predictions are made on the whole dataset.

4-Errors are calculated by comparing the predictions and actual values.

5-While creating the next model, higher weights are given to the data points which were predicted incorrectly.

6-Weights can be determined using the error value. For instance, higher the error more is the weight assigned to the observation.

7-This process is repeated until the error function does not change, or the maximum limit of the number of estimators is reached.



### Stacking

Stacking is a similar to boosting:

you also apply several models to your original data. The difference here is, however, that you don't have just an empirical formula for your weight function, rather you introduce a meta-level and use another model/approach to estimate the input together with outputs of every model to estimate the weights or, in other words, to determine what models perform well and what badly given these input data. and finally I get its true illustration.

#  🍲 ModelSoups
Авторы статьи “[Model soups: averaging weights of multiple fine-tuned models improves accuracy without increasing inference time](https://arxiv.org/abs/2203.05482)” описали свой подход и провели эксперименты по усреднению весов моделей. Сделано это было для создания единственной модели с наилучшими характеристиками по качеству и устойчивости.

**Идеальная техника ансамблирования** с точки зрения затрат на вычислительные ресурсы — техника, которая позволила бы получить идентичные ансамблю свойства при количестве вычислений, равным одной модели. Есть способ решения подобной задачи: усреднение весов множества моделей. В рамках этого направления исследований можно выделить общий подход:

1. Используем много обученных моделей или чекпоинты, полученные в рамках одного обучения на разных эпохах.
2. Подбираем множество комплиментарных моделей (при усреднении их весов получается наилучшая модель).
3. Выбираем и применяем один из алгоритмов усреднения.

После усреднения у полученной модели улучшаются следующие характеристики:

1. Качество работы на валидационных данных.
2. Устойчивость к сдвигам в распределении и out-of-distribution примерам.

## Как варим?

### Задача

Начнем с постановки задачи. Авторы предлагают представить базовый процесс обучения модели, который чаще всего применяют при решении кастомных задач:

1. Находим интересующую нас архитектуру и чекпоинт, полученный в режиме supervised обучения на одном из больших наборов данных / в self-supervised режиме.
2. Меняем финальный линейный слой модели на слой, подходящий к задаче.
3. Определяем набор гиперпараметров обучения — настройки оптимизатора, планировщика LR , а также подходящий под задачу набор аугментаций.
4. Делаем finetuning под нашу задачу. Можно дообучить только веса линейного слоя, заморозив прочие параметры, или же произвести end-to-end дообучение модели с настройкой всех имеющихся параметров.

Как правило, при обучении происходит конфигурация множества параметров, влияющих на результаты. Если архитектура и датасет зафиксированы — на начальном этапе варьируются следующие параметры:

- выбор LR, weight decay, параметров планировщика LR;
- выбор набора аугментаций.

### Дизайн эксперимента

Для проверки данной гипотезы был составлен следующий дизайн эксперимента:

1. В качестве **предтренированной модели** авторы берут CLIP ViT-B/32 (OpenAI).
2. В качестве **целевого набор данных** — ImageNet.
3. В качестве **наборов данных для проверки устойчивости** — ImageNet-V2, R, Scetch, A и набор данных ObjectNet.
4. Также используют **end-to-end** как ****вариант дообучения (настраивая все имеющиеся параметры).
5. Устанавливают **случайный seed** и **порядок данных**.
6. Семплируют **случайные значения гиперпараметров** для:
    - LR;
    - Weight decay;
    - Augmentation strength;
    - Mixup;
    - Label smoothing.

Чуть позже мы поговорим о влиянии значений гиперпараметров на качество финальной модели после усреднения. А сейчас рассмотрим методы усреднения от авторов статьи. Каждый такой метод лаконично назван **Recipe** (рецепт).

### Рецепты, или алгоритмы усреднения весов моделей

Авторы во время основных экспериментов используют следующий набор рецептов:

- **Uniform soup** $f(x,\frac{1}{k}\sum^{k}_{i=1}\theta_{i})$ — average весов всех полученных моделей;
- **Greedy soup** $\textbf{Recipe 1}$ — выборочное усреднение;
- **Learned soup** $\textbf{Recipe 2}$ — усреднение по выученным коэффициентам.

$\textbf{Recipe 1 : GreedySoup}$

$\textbf{Input}: \text{Набор чекпоинтов (ингредиентов супа)}:\ \{{\theta_{1}}, ..., {\theta_{k}}\}$

$\text{где}\ \{{\theta_{1}}, ..., {\theta_{k}}\} - \text{отсортированы по убыванию точности на валидации}$ $\text{ValAcc}({\theta_{i}})$

$\text{ingredients} ← \{\} - \text{финальный набор чекпоинтов}$

$\textbf{for}\ \text{i = 1}\ \textbf{to}\ \text{k}\ \textbf{do} :$
${if} \space {ValAcc}({average}({ingredients} ∪ {\theta_{i}})) ≥ {ValAcc}({average}({ingredients}))$
${then}\space {ingredients} ← {ingredients} ∪ {\theta_{i}}$ 

$\textbf{return}\ \text{average}(\text{ingredients})$

**Жадный алгоритм**, предложенный авторами, основан на переборе всех имеющихся чекпоинтов в порядке убывания их точности. То есть мы начинаем с наилучшего чекпоинта и итеративно усредняем и подсчитываем точности на валидационной выборке. Оставляем только те чекпоинты, которые улучшают показатель точности при добавлении в итоговую комбинацию.

**Преимущества алгоритма:**

- он проверяет все полученные чекпоинты;
- чекпоинты, которые ухудшают итоговый результат, не попадают в “суп”: минимальный результат по точности итоговой модели равен точности наилучшей модели после обучения.

**Недостатки алгоритма:**

- стратегия выбора субоптимальна, ведь мы не исследуем другие последовательности проверки чекпоинтов, кроме перебора по уменьшению точности;
- чекпоинты, при добавлении которых точность финальной модели растет, могут и не улучшить точность на out-of-distribution данных.


$\textbf{Recipe 2 : LearnedSoup}$

$\textbf{Input}: \text{Набор чекпоинтов (ингредиентов супа)}:\ \{{\theta_{1}}, ..., {\theta_{k}}\}$

$\text{Обучаемый вектор весов}: W^{1\times k}_{soup}, \beta - \text{Параметр температуры}, \\E - \text{количество итераций}$

${(x_{j}, y_{j})}^n_{j=1} - \text{ валидационный датасет}$

$\textbf{for}\ \text{e = 1}\ \textbf{to}\ \text{E}\ \textbf{do} :$

$\alpha^{e} = softmax(W^{e}_{soup})$ — весовые коэффициенты линейной комбинации моделей

$\mathit{scores} = f(x_{j},\sum^{k}_{i=1}\alpha^e_{i}\theta_{i})$ — ответы модели, полученной с весами $\alpha^{e}$

$l^e = \sum^n_{j=1}L(β · \mathit{scores},y_{j})$ — ошибка на всех примерах валидационного датасета

$W^{e+1}_{soup} = {W^{e}_{soup}} -\eta\frac{\partial l^e}{\partial W^{e}_{soup}}$ , где $\eta$ — скорость обучения

При подборе параметров $W^{e}_{soup}$ параметры $\{{\theta_{1}}, ..., {\theta_{k}}\}$ заморожены. Еще авторы сообщают: в качестве альтернативного подхода можно подбирать веса не для всей модели, а для каждого отдельного слоя.

**Стоит отметить:** подобный метод очень похож на подбор температуры во время калибровки модели при работе с ансамблями.


### Справка по рецептам
- **Uniform soup** гарантирует наилучшую устойчивость среди рецептов, но ухудшает качество на основной задаче;
- **Greedy soup** дает 2 результат с наилучшей последовательностью выбора чекпоинтов;
- Порядок выбора моделей в **Greedy soup** влияет на итоговый результат со значением дисперсии в 0.05 и 0.16 соответственно;
- Наилучшие результаты получаются с помощью **Learned Soup** (послойного), но он требует подбора NxM параметров, где N — количество моделей кандидатов, а M — количество слоев.
### Почему работает?
В статье “[Taxonomizing local versus global structure in neural network loss landscapes](https://arxiv.org/abs/2107.11228)” авторы показывают различные типы локальных минимумов. Дизайн эксперимента (архитектура, количество данных, гиперпараметры и др.) здесь обуславливает попадание в них.

![[Pasted image 20240408155636.png]]
Подобные локальные минимумы в статье Model Soups авторы называют **Basin (впадина).**

> **Basin** — локальное пространство ландшафта функции потерь, в которое мы попадаем в процессе стохастической оптимизации.

Есть вариации локальных минимумов:

1. **Globally poorly-connected**
    1. **Phase I** — высокие значения ошибки, в окрестности текущей точки значение ошибки сильно колеблется, различные базины плохо связаны;
    2. **Phase III** — низкие значения ошибки, колебания в окрестности текущей точки несущественны, различные базины все еще плохо связаны.
2. **Globally well-connected**
    1. **Phase II** — высокие значения ошибки, колебания в окрестности текущей точки присутствуют, различные базины связаны, но на пути встречается флуктуация значения ошибки;
    2. **Phase IV-A,B** — низкие значения ошибки, колебания в окрестности текущей точки несущественны, различные базины хорошо связаны между собой.
Авторы Model Soups отмечают: их подход работает, если все модели лежат в рамках одной базины (графически это похоже на варианты Phase IV-A и Phase IV-B)

Чтобы разобраться, как в пространстве функции потерь располагается итоговая точка после обучения (определенная комбинация значений весов нейронной сети), обратимся к результатам двух работ: “[Deep Ensembles: A Loss Landscape Perspective](https://arxiv.org/abs/1912.02757)” и “[Deep learning versus kernel learning: an empirical study of loss landscape geometry and the time evolution of the Neural Tangent Kernel](https://arxiv.org/abs/2010.15110)”.

В первой работе авторы рассматривают **динамику обучения моделей** при разной начальной инициализации и **размещение итоговых точек весов** в функциональном пространстве.

Авторы запускают обучение модели с разной начальной инициализацией
При запуске с различной инициализацией обе модели сходятся к идентичным(практически) значениям ошибки, но с точки зрения функционального пространства полученные функции отличаются. Случайная инициализация приводит к расположению модели в разных базинах. К таким же результатам пришли и в статье “[Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs](https://arxiv.org/abs/1802.10026)”, где провели 3 независимых запуска обучения модели со случайной начальной инициализацией. Визуализировав полученное пространство, авторы получили следующий результат:
![[Pasted image 20240408155952.png]]Результат обучения каждой индивидуальной модели приводит к схожим значениям ошибки, но с точки зрения расположения мы находимся в различных базинах. Исходя из этого можно сделать следующий вывод: при **работе с методом Model Soups необходимо использовать идентичную начальную инициализацию, иначе мы не получим нужного результата при усреднении весов моделей.**

Теперь рассмотрим результаты работы “[Deep learning versus kernel learning: an empirical study of loss landscape geometry and the time evolution of the Neural Tangent Kernel](https://arxiv.org/abs/2010.15110)”. В ней авторы исследуют эволюцию нейронной сети с помощью Neural Tangent Kernel. Помимо экспериментов со случайной инициализацией параметров авторы также рассматривают режим, при котором у нескольких нейронных сетей идентична начальная инициализация. Они представляют результаты экспериментов следующим изображением
![[Pasted image 20240408160600.png]]Мы видим два симплекса с расположенными на них точками. Границы симплекса (черная линия) — регионы с низким значением ошибки, белым цветом указан регион с высоким значением ошибки.

**Эксперимент A**

Авторы запускают два обучения одной и той же модели, но с различной инициализацией. Результаты обучения — красные точки на границах симплекса. Оранжевым цветом показан линейный путь от одной модели к другой. Как и в ранее рассмотренных экспериментах, такое соединение проходит через пространство высокого значения ошибки, поэтому для соединения весов двух моделей необходимо использовать нелинейный путь (желтый цвет).

**Эксперимент B**
Авторы запускают обучение в режиме, похожем на Model Soups. Они стартуют от синей точки и семплируют на различных итерациях обучения по две модели в зеленых точках. Семплированные модели доучиваются до сходимости, их обозначают красными точками на симплексе. Как показано на изображении выше, для ранней итерации создания моделей (первая зеленая точка) они, как и в эксперименте А, находятся на разных границах симплекса. Но если семплирование произошло позже — полученные точки будут находиться на одной границе симплекса. Тогда между ними можно сделать линейную интерполяцию, при которой значение ошибки во всех точках на пути будет низким.

Изучив теоретические исследования, мы приходим к следующим выводам:

- нужна **идентичная начальная инициализация** (в целом следуем подходу из Model Soups);
- нужна **достаточно обученная модель**, с которой мы инициализируемся (некоторые эмпирические результаты по этому вопросу рассмотрим далее);
- для получения успешных результатов при усреднении весов моделей также нужна настройка **параметров обучения**, ведь они влияют на те минимумы, в которые попадет модель.

### Mode connectivity and Linear mode connectivity

**Mode connectivity**

В рамках работы с глубокими нейронными сетями анализ свойств связанности весов различных моделей был представлен в статье ["Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs"](https://arxiv.org/abs/1802.10026) и ["Essentially No Barriers in Neural Network Energy Landscape"](https://arxiv.org/abs/1803.00885). Здесь авторы предложили методы для поиска пути между независимо обученными нейронными сетями, находящимися в разных базинах. Изначальная постановка задачи требовала реализации такого алгоритма поиска пути, при котором на всем его протяжении значение мат ожидания ошибки будет минимальным. Используя Bezier и Polychain кривую, авторы за счет процесса смещения весов вдоль этих кривых приходят к следующим результатам (подробнее об этом — в [видео](https://youtu.be/37wntPh_24Y?si=ZzoPlMxcTdrDJ8ZR&t=1649) от автора статьи):
![[Pasted image 20240408161602.png]]
Результаты двух работ отлично применимы для быстрого построения ансамблей. Подробнее об использовании таких путей между весами и для построения ансамблей можно прочитать в статье **“[Learning Neural Network Subspaces](https://arxiv.org/abs/2102.10472)”.**

При всем успехе данных методов для работы Model Soups нужна линейная связанность весов. Поэтому давайте обратимся к **Linear mode connectivity.**


**Linear mode connectivity** 
А если мы инициализировали все модели идентично (как, например, в Model Soups) и захотели понять, сходятся ли они в одну базину или принадлежат к разным? Авторы статьи ["Linear Mode Connectivity and the Lottery Ticket Hypothesis"](https://arxiv.org/abs/1912.05671) формулируют задачу анализа подобных свойств следующим образом:

**Пусть нам дано:**

- $\mathit{N}$ — модель
- $\mathit{W}$ — инициализированные веса модели
- $\mathit{SGD}$ — оптимизатор
- $\mathit{U}$ — распределение шума (сюда входит набор агументации, последовательность батчей и другие параметры, которые мы можем задать случайно)

**Задаем вопрос:**

Какова устойчивость $\mathit{SGD}$ к случайному шуму, семплированному из $\mathit{U}$ ? Здесь под **устойчивостью** подразумевается отсутствие возрастания значений функции потерь при интерполяции между весами $\mathit{W}_1$ и $\mathit{W}_2$, полученными при идентичной инициализации, но с разными семплами шума из $\mathit{U}$.

$$
\mathcal{E_{a}}(W_{1}, W_{2}) = \mathcal{E}(aW_{1} + (1−a)W_{2}) \\ \space где \space \\ W_{1}, W_{2} - веса\ полученных\ моделей \\ \space  \mathcal{E} - значение\ ошибки\\ \space  a \in [0,1] - коэффициент\ интерполяции
$$

Рассмотрим эксперименты авторов, которые проливают свет на границы работоспособности Model Soups.
1. Результаты обучения моделей с 0: у них идентичная случайная инциализация.
	1.Согласно результатам исследования получаем следующее:
	- для сети LeNet и датасета MNIST оптимизация с SGD является устойчивой к различным шумам и приводит к идентичной базине;
	- для всех остальных наборов данных и нейронных сетей обучение с общей случайной инициализацией сводится к различным базинам.
2. за начальные веса $\mathit{W}$ берется чекпоинт после k итераций обучения, и каждая из сетей доучивается независимо. Результаты следующие:
	1. Согласно результатам исследования получаем следующее:
		- для сети LeNet и датасета MNIST значение k не влияет на устойчивость к шуму;
		- для набора данных CIFAR-10 и сетей ResNet-20 и VGG-16 устойчивость возникает, если модели инциализируются с весов при k ≥ 2000 для ResNet и k≥1000 для VGG (что эквивалентно 3 и 1.5 процентов от всего обучения при batch size = 128);
		- для набора данных ImageNet для сети ResNet-50 устойчивость возникает с 18 эпохи (20% от всего обучения (90 эпох) при batch size = 1024), для Inception-v3 — с 28 эпохи (16%, количество эпох равно 171).
Результаты экспериментов показывают, на какой итерации модель переходит из “early chaotic” к “late stable”.


Исходя из полученных эмпирических результатов, можно сделать следующие выводы, которые позволяют определить требования к инициализации в Model Soups:

1. Для крошечных наборов данных (MNIST) даже случайная инициализация позволит получить линейно связанные веса.
2. С ростом количества примеров в наборе данных растет и требуемое количество эпох обучения (начиная от 3% для CIFAR-10 и до 20% на ImageNet).

Теперь, обладая знаниями о свойствах работы Model Soups, взглянем на известный метод **SWA** и поймем, в чем связь и отличие данных методов (ранее у нас выходила статья [Weight Averaging](https://www.notion.so/07a2c80c6cf54858887d71fac70d8a31?pvs=21) про базовое описание **SWA** и имплементацию в коде).

### Model Soups VS Stohastic Weight Averaging. А в чем разница?
Кратко суммируем [SWA](https://arxiv.org/abs/1803.05407). Перед стартом нам понадобится:

- $\mathit{N}$ — модель
- $\mathit{W}$ — проинициализированные веса модели
- $\mathit{SGD}$ — оптимизатор
- $\mathit{Cycle\ lenght}$ — длина цикла при циклическом расписании LR (при константном = 1)
- $\mathit{E_n}$ — количество эпох дообучения

Именно с ранее предобученной модели (75% от времени тренировки для CIFAR-10 и СIFAR-100 , для Imagenet с предтренированного чекпоинта из torchvision) авторы начинают процесс SWA. Проиллюстрировать два разных варианта его настройки можно следующей схемой:

![[Pasted image 20240408162712.png]]

При каждом окончании цикла (на изображении он отмечен оранжевыми точками) авторы применяют следующую формулу для усреднения весов:
$$
W_{\mathit{SWA}} ← \frac{W_{\mathit{SWA}}·n_{\mathit{models}}+W}{ n_\mathit{models}+1}
$$
$где\ W - веса\ после\ градиентного\ шага\ в\ точке\ окончания\ цикла\ n_{\mathit{models}} - количество\ использованных\ чекпоинтов\ \ W_{\mathit{SWA}} - текущий\ результат\ усреднения$

**Важно отметить**: поскольку в качестве начальной инициализации используются веса, полученные после достаточного по времени начального обучения, мы можем считать, что находимся в одной базине. Фактически особый вид расписания скорости обучения позволяет эффективно исследовать окрестность этой базины.

Давайте кратко рассмотрим свойства циклической скорости обучения — наиболее часто применяемой скорости при запуске SWA.

**Циклическое расписание скорости обучения** позволяет эффективно исследовать ландшафт функции потерь как локально (в рамках одной базины), так и более глобально (при поиске новых базин).
![[Pasted image 20240408163020.png]]Как видно из рисунка выше (визуализация правой части), изменение скорости обучения в начале каждого нового цикла приводит к переходу в другое пространство ландшафта функции потерь (в другую базину) и до конца цикла, пока скорость обучения снижается, модель постепенно сходится к минимуму этой базины (изменение ошибки показано направлением стрелки). С помощью такого приема можно получать множество независимых чекпоинтов для формирования ансамбля, как это сделали авторы “[Loss Surfaces, Mode Connectivity, and Fast Ensembling of DNNs](https://arxiv.org/abs/1802.10026)” и "[Snapshot Ensembles: Train 1, get M for free](https://arxiv.org/abs/1704.00109)”. После анализа связанности данных весов через Linear mode connectivity авторы статьи “[Exploring loss function topology with cyclical learning rates](https://arxiv.org/abs/1702.04283)”. получили следующий результат:

![[Pasted image 20240408163107.png]]При стандартном обучении на интервале от 0 до 1 наблюдается консистентное значение ошибки. В режиме циклической тренировки мы имеем резкое возрастание ошибки при a = 0.5, что сигнализирует о нахождении весов в разных базинах. Тут может возникнуть вопрос: почему при усреднении по таким чекпоинтам в SWA у нас получается итоговый рост в качестве? Есть ряд ключевых компонентов, которые позволяют перейти из глобального режима в локальный:
- старт с модели, обученной достаточное количество эпох;
- короткий цикл — в SWA используется цикл из 5 эпох, а в SSE — от 20 до 50 эпох (в зависимости от нейронной сети);
- значение LR для цикла — наилучшие результаты SWA достигает при значениях максимального и минимального LR = 5 · 10−2 и 5 · 10−4 соответственно, тогда как в Snapshot Ensembles используется 1 * 10-1 в качестве максимального.
Как видно из рисунка выше, значение LR существенно влияет на результаты SWA.

Итак, остановимся на финальном сравнении SWA и Model Soups:

**Model Soups**

- стартуем с модели, обученной достаточное количество эпох
- важно, чтобы начальная модель достаточно сошлась
- N независимых дообучений
- исследование базины за счет случайного шума (порядок батчей, аугментации, значение LR)
- есть ограничение на значения гиперпараметров

**SWA**

- стартуем с модели, обученной достаточное количество эпох
- важно, чтобы начальная модель достаточно сошлась
- одно долгое дообучение
- исследование базины за счет варьирования LR
- есть ограничение на значения гиперпараметров

Финальный вопрос: как выбрать скорость обучения для Model Soups? Для ответа обратимся к результатам статьи “[To Stay or Not to Stay in the Pre-train Basin: Insights on Ensembling in Transfer Learning](https://arxiv.org/abs/2303.03374)”, авторы которой провели множество реальных экспериментов как по анализу ансамблей в режиме тренировки с предварительным обучением модели, так и по анализу работы Model Soups. Они помогут понять, какой режим тренировки будет оптимальным для получения наилучших результатов при усреднении весов моделей.

### Исследуем пространство базины: влияние настроек обучения
Начнем с краткого определения задачи. Авторы формулируют ее в виде следующего вопроса: какой подход к дообучению нейронной сети нужно применить, чтобы из N дообученных сетей получить наилучшее качество ансамбля моделей? При этом предполагается найти такой подход, который мог бы позволить использовать минимальное количество GPU часов, потраченных на обучение подобного ансамбля. Авторы рассматривают два подхода к получению подобных моделей (см. Рисунок 22):

1. **SSE** — равен варианту из SWA и Snapshot Ensebles (применяем циклическое расписание LR для дообучения модели, чекпоинты сохраняются в конце каждого цикла);
2. **StarSSE** — для каждой модели используется свой независимый запуск дообучения. Авторы характеризуют его как параллельную версию SSE. Этот вариант максимально схож с Model Soups.

![[Pasted image 20240408163612.png]]
Переходим к практическим результатам метода Model Soups и получаем следующее:
![[Pasted image 20240408163631.png]]
При использовании параллельного циклического дообучения (StarSSE) итоговое усреднение дает наилучшую точность, и все полученные модели улучшают итоговую точность.
- **Local DE soup** — Model Soups из моделей, обученных при случайной инициализации последнего (линейного) слоя с идентичными гиперпараметрами.
- **Soup size** — ****количество индивидуальных моделей, усредненных между собой с помощью Uniform soup.

На графиках мы видим расположения полученных весов и значения функции потерь для них. Эксперименты с большими циклами и высоким LR дают модели, расположенные на границе базины **(semi-local).** Их точность хуже тех, где использовались маленькие **(local)** и оптимальные **(optimal)** значения параметров LR.

Результаты работы добавляют следующие пункты в пайплайн использования Model Soups:

- для получения оптимальных чекпоинтов нужно использовать OneCycle с начальным значением LR от x2 до x4 при снижении количества эпох до 0.25 от их количества;
- Star-SSE можно использовать как drop-in замену SWA в пайплайне дообучения и получать лучшие результаты;
- авторы не варьировали параметры аугментаций, mixup-добавление этих переменных вместе со Star-SSE может дополнительно улучшить результат Model Soups.

### Промежуточные выводы
Итак, давайте закрепим **финальные выводы по разделу:**

1. Случайная инициализация приводит модели в разные участки функционального пространства. Их не получится усреднить.
2. Для получения весов, подходящих Model Soups, нужна общая начальная инциализация.
3. Эта инициализация должна быть обучена достаточное количество эпох (здесь все зависит от размера сети и сложности датасета).
4. Модели, подходящие Model Soups, — это модели в одной базине. При линейной интерполяции между ними не должно возникать существенного увеличения значения ошибки во всех точка интерполяции.
5. SWA — аналог Model Soups, но с другим подходом к исследованию базины.
6. Для получения оптимальных чекпоинтов необходимо использовать OneCycle LR sheduler с начальным значением LR от x2 до x4 от базового, снизив при этом количество эпох обучения (минимально до 0.25 от начального).
7. Применение StarSSE вместе с вариацией пайплайна аугментаций может быть бейзлайном при построении Model Soups.

### Использование Model Soups в различных задачах
####  Улучшение робастности при файнтюнинге foundation models

Для знакомства с этой задачей можно почитать две статьи: **“[Robust fine-tuning of zero-shot models](https://arxiv.org/abs/2109.01903)”** и **“[Patching open-vocabulary models by interpolating weights](https://arxiv.org/abs/2208.05592)”**. Мы кратко рассмотрим первую из них.

**Основная идея методов**: получить более **робастную модель** за счет **усреднения весов начальной (zero-shot) и дообученной на конкретной задаче модели**. Поскольку у модели уже есть неплохая точность из коробки, а дообучение двигает ее веса только в рамках базины — между такими моделями есть линейная связь. Давайте рассмотрим, как авторы используют подобные свойства в методе **Robust fine-tuning of zero-shot models**.

Формулируют они задачу так: пусть в качестве in distribution набора данных выступает ImageNet, а в качестве distribution shift данных (со специальными сдвигами в распределении) — наборы данных, идентичные Model Soups.

**Вопрос**: “Может ли усреднение весов начальной и дообученной модели дать наилучшую устойчивость на distribution shift при хорошем качестве на in distribution?”.

Анализ результатов **пайплайна**:

1. В качестве начальной модели берется **CLIP.**
2. Модель файнтюнится на **ImageNet** (в end-to-end режиме).
3. С помощью выражения $\mathcal{\theta_{a}} = (1-a) *\theta_{zero-shot} + a * \theta_{fine-tuned}$ и выбранного значения коэффициента $a$ формируется итоговая модель.
4. Далее измеряется точность полученной модели in distribution и distribution shift данных.
![[Pasted image 20240408172255.png]]Обозначения на графике выше:

- оранжевый кубик — значение точности дообученной CLIP модели в end-to-end режиме;
- синяя линия — точность всевозможных supervised моделей на ImageNet;
- фиолетовая линия — точность всевозможных CLIP моделей, где дообучались только веса линейного слоя;
- розовая линия — результаты усреднения при различном коэффициенте $a$.

Мы видим: для всевозможных CLIP моделей при дообучении только линейного слоя точность может быть выше, однако начальная робастность к distribution shift ниже в сравнении с end-to-end дообучением.

**Результаты Model Soups при различном коэффициенте $a$**:

- при небольших значениях $a$ качество растет на всех датасетах;
- при значении $a$, близком к середине, получается наилучшее distribution shift качество при идентичном in distibution.

#### Возможность гибко настраивать поведение LLM после стадии RLHF

Современный **пайплайн обучения LLM** предполагает две основных стадии:

1. Начальную предтренировку на огромных наборах данных из интернета.
2. Дообучение модели с помощью RL в задаче выполнения инструкций.

На втором шаге обучения наблюдается закономерность: чем лучше ответы модели соответствуют человеческим предпочтениям, тем больше итоговая награда. **Стоит отметить:** эта стадия “выравнивания” предсказаний модели к нашему представлению о поведении персонализированного ассистента, как правило, требует очень масштабной работы с данными и учета в них множества факторов.

Следовательно, возникает следующая **проблема**: если на стадии RLHF мы не учли что-то в рамках функции вознаграждения — для части пользователей поведение модели и ее ответы не будут оптимальными. Для возможности более гибкой настройки модели после обучения авторы **“[Personalized Soups: Personalized Large Language Model Alignment via Post-hoc Parameter Merging](https://arxiv.org/abs/2310.11564)”** придумали свой подход: за счет усреднения весов после обучения с помощью Model Soups мы решаем задачу более тонкой настройки модели согласно предпочтениям конкретного пользователя. Схематично это выглядит так:

![[Pasted image 20240408172944.png]]
**Идея метода:**

1. Сначала делаем процедуру RLHF, формулируя Reward так, чтобы итоговое поведение модели удовлетворяло среднему пользователю на планете, получаем General веса модели.
2. Далее под каждый интересующий нас вариант поведения формулируем особый Reward и также получаем финальные (персонализированные) веса.
3. В зависимости от требований пользователя усредняем веса разных персонализированных моделей и General, используя при этом весовые коэффициенты для контроля вклада каждой модели.

Таким образом, мы получаем удобное Post-hoc решение, при котором можем расширять базу персонализированных весов и добавлять больше различных настроек для персонализации модели под пользователя.

#### Адаптация LLM под новую задачу без обучения

Сегодня для тюнинга LLM и других гигантских моделей под конкретную пользовательскую задачу наиболее эффективно используется **LoRA** для обучения только небольшой добавки к существующим весам модели.

Применение LoRA не сдвигает веса модели существенно: они остаются в той базине, где модель сошлась после начального дообучения. Следовательно, если у нас есть много независимо обученных LoRA — выученные добавки к весам можно будет усреднить и получить модель, которая решает новую задачу лучше начальной модели.

Важно отметить: речь идет именно про zero-shot работу модели, ведь использование множества примеров “запрос-ответ” внутри входного промта — наиболее простое решение подобной задачи. Однако оно требует постоянного процессинга дополнительных токенов перед запросом пользователя.

Подобная парадигма работы со множеством ранее обученных адаптеров представлена в работе **“[LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition](https://arxiv.org/abs/2307.13269)”.**

Авторы предлагают новую парадигму адаптации LLM без обучения:

![[Pasted image 20240408173407.png]]Авторы формулируют **начальные условия**:

1. Есть заранее выбранная **LLM** для решения задачи;
2. Есть хаб с множеством уникальных **LoRA параметров**, полученных при дообучении под различные датасеты;
3. Есть набор **пар “запрос-ответ”** в качестве примеров запросов на инференсе модели.

**Задача**: использовать множество уникальных LoRA параметров для решения новой задачи.

**Алгоритм решения:**

1. Создадим вектор весовых коэффициентов под каждый из адаптеров;
2. Усредним все адаптеры с заданными весовыми коэффициентами (аналогично Model Soups);
3. Добавим полученную добавку к весам модели;
4. Оценим ошибку модели при текущих весах для каждого адаптера.

![[Pasted image 20240408173517.png]]
Затем мы повторяем все шаги и изменяем при этом вектор весов таким образом, чтобы достичь наименьшей ошибки на имеющихся у нас примерах. Процесс похож на Learned Soup (о нем мы рассказывали в первом разделе), за исключением проведения оптимизации с помощью неградиентного метода ([https://facebookresearch.github.io/nevergrad/](https://facebookresearch.github.io/nevergrad/)).

#### Улучшение доменной генерализации при семантической сегментации
Идея получить модель, которая будет хорошо работать сразу с несколькими доменами, не нова: в этом направлении каждый год появляются разные методы. В статье “[A Re-Parameterized Vision Transformer (ReVT) for Domain-Generalized Semantic Segmentation](https://arxiv.org/abs/2308.13331)” представлен подход, где Model Soups используется для получения модели, хорошо работающей сразу на нескольких набора данных.

![[Pasted image 20240408173631.png]]
**Идея метода:** пусть у нас есть начальный набор данных и набор целевых датасетов, на которых мы бы хотели видеть модель с высоким качеством. В частности мы предполагаем, что начальный и целевой датасеты относятся к одному большому домену. **Задача** сводится к предложению такого метода обучения на начальном наборе данных, при котором качество будет расти и на целевом наборе. Авторы предложили свой **дизайн эксперимента:**

- выбираем сеть для сегментации (в экспериментах авторов это **Segformer B2, B3, B5**);
- берем в качестве начального набора данных **синтетический датасет GTA5** для задачи семантической сегментации в домене автономных машин;
- берем в качестве целевых наборов данных для валидации модели **Cityscapes Mapillary Vistas, BDD100k, ACDC, KITTI** из того же общего домена.

**Обучение модели по следующему сценарию:**

1. Инициализируем энкодер весами модели, предтренированной на ImageNet.
2. Инициализируем декодер случайным образом.
3. Используем специфический набор аугментаций под каждую отдельную модель.

**Пайплайн аугментаций** выглядит так:
![[Pasted image 20240408173756.png]]
Фактически модели получают разный по силе **набор аугментаций:**

- **базовые** — Resize, Random Crop, Flip;
- **усиленные** — PhotoAug, Bilateral Filter;
- **наиболее сильная аугментация** — PixMix.

Подробнее о настройках обучения и аугментаций можно посмотреть в Supplementary Material, расположенном после источников в статье на [arxiv.](https://arxiv.org/abs/2308.13331)

**Пайплайн для усреднения** следующий:

1. Выбираем три модели комбинации, которые дают наилучшее качество;
2. Усредняем их;
3. В качестве декодера выбираем один из имеющихся согласно метрикам.

#### Применение в неструктурированном прунинге
В рамках этой задачи Model Soups используют для улучшения точности спарсифицированной модели. В статье “[Sparse Model Soups: A Recipe for Improved Pruning via Model Averaging](https://arxiv.org/abs/2306.16788)” авторы предложили применять усреднение весов моделей для улучшения результатов процесса неструктурированного прунинга. Кстати, о прунинге мы писали в одной из наших [статей](/e5776c9b167b4ce8af4298a0d4db2497?pvs=25) 😊

Для прунинга авторы применили алгоритм **Iterative Magnitude Pruning (IMP)**. Он состоит из следующих шагов:

1. Начинаем с ранее обученной модели;
2. Для всех весов в каждом из сверточных / линейных слоев вычисляем L1 норму;
3. Зануляем заранее заданный % наименьших по L1 норме весов для каждого из слоев;
4. Переобуваем сеть.

Весь процесс визуально выглядит так:
![[Pasted image 20240408174335.png]]
Авторы предложили для алгоритма дополнение в виде Model Soups — оно позволило улучшить итоговые результаты прунинга. Идея следующая: производить не одно дообучение модели, а несколько параллельных, стартующих от единой инициализации, но с разными настройками обучения. Это повторяет процесс Model Soups. По завершению обучения веса моделей усредняются.
![[Pasted image 20240408174409.png]]

# KAN: Kolmogorov–Arnold Networks

While MLPs have fixed activation functions on nodes (“neurons”), KANs have learnable activation functions on edges (“weights”). **KANs have no linear weights at all – every weight parameter is replaced by a univariate function parametrized as a spline.**

y. For accuracy, much smaller KANs can achieve comparable or better accuracy than much larger MLPs in data fitting and PDE solving. Theoretically and empirically, KANs possess faster neural scaling laws than MLPs. For interpretability, KANs can be intuitively visualized and can easily interact with human users.
![[Pasted image 20240502142437.png]]
## Kolmogorov-Arnold Representation theorem
Vladimir Arnold and Andrey Kolmogorov established that if f is a multivariate continuous function on a bounded domain, then f can be written as a finite composition of continuous functions of a single variable and the binary operation of addition.

Eq. (2.1):
![[Pasted image 20240502143851.png]]
In a sense, they showed that the only true multivariate function is addition, since every other function can be written using univariate functions and sum. One might naively consider this great news for machine learning: learning a high-dimensional function boils down to learning a polynomial number of 1D functions. However, these 1D functions can be non-smooth and even fractal, so they may not be learnable in practice.

However, we are more optimistic about the usefulness of the Kolmogorov-Arnold theorem for machine learning. First of all, we need not stick to the **original  which has only two-layer nonlinearities and a small number of terms (2n + 1) in the hidden layer**: we will generalize the network to arbitrary widths and depths. Secondly, most functions in science and daily life are often smooth and have sparse compositional structures, potentially facilitating smooth Kolmogorov-Arnold representations. **The philosophy here is close to the mindset of physicists, who often care more about typical cases rather than worst cases. After all, our physical world and machine learning tasks must have structures to make physics and machine learning useful or generalizable at all**
## KAN architecture 

Suppose we have a supervised learning task consisting of input-output pairs {xi , yi}, where we want to find f such that yi ≈ f(xi) for all data points. Eq. (2.1) implies that we are done if we can find appropriate univariate functions ϕ_q,p and Φ_q. This inspires us to design a neural network which explicitly parametrizes Eq. (2.1).

**Since all functions to be learned are univariate functions, we can parametrize each 1D function as a B-spline curve, with learnable coefficients of local B-spline basis functions**
Now we have a prototype of KAN, whose computation graph is exactly specified by Eq. (2.1)  with the input dimension n = 2, appearing as a two-layer neural network with activation functions placed on edges instead of nodes (simple summation is performed on nodes), and with width 2n + 1 in the middle layer.
The breakthrough occurs when we notice the analogy between MLPs and KANs. In MLPs, once we define a layer (which is composed of a linear transformation and nonlinearties), we can stack more layers to make the network deeper. To build deep KANs, we should first answer: “what is a KAN layer?” It turns out that a KAN layer with nin-dimensional inputs and nout-dimensional outputs can be defined as a matrix of 1D functions
![[Pasted image 20240502150139.png]]
**Implementation details**
![[Pasted image 20240502152925.png]]![[Pasted image 20240502153054.png]]
Then there are in total O(N^2L(G + k)) ∼ O(N^2LG) parameters. In contrast, an MLP with depth L and width N only needs O(N^2L) parameters, which appears to be more efficient than KAN. Fortunately, KANs usually require much smaller N than MLPs, which not only saves parameters, but also achieves better generalization and facilitates interpretability. 




# Self-supervised learning

Нам понадобится 3 основных термина:

- **предварительная задача (pretext task)** — сама задача SSL с искусственной разметкой. Именно ее решает модель, чтобы научиться извлекать хорошие признаки из данных;
- **псевдо-разметка** — та самая дешевая разметка, происходящая без участия человека;
- **последующая задача (downstream task)** — задача, по которой проверяют качество выученных признаков. Как правило, это простые модели (KNN, LinReg, LogReg и другие), обучающиеся на извлекаемых с помощью SSL-модели признаках. Иногда бывает и так, что модель не фиксируется и дообучается целиком.
![[Pasted image 20240506144015.png]]
**Виды предварительных задач**:
В современных SOTA-методах процедура генерации псевдо-разметки в основном сводится к двум вариантам:
1. Multi-view invariance: здесь псевдо-разметка формируется по принципу contrastive learning, то есть позитивными примерами являются два по-разному аугментированных варианта одного и того же изображения, а негативными — аугментированные варианты другого изображения. ![[Pasted image 20240506144130.png]]
2. Задача восстановления информации: из изображения удаляются некоторые патчи, а сеть учится восстанавливать эту информацию. В таком случае псевдо-разметка состоит из пар (X, Y), где Y — исходное изображение, а Х — маскированная версия этого изображения, в котором удалена часть информации. ![[Pasted image 20240506144154.png]]
**Выделяют четыре больших семейства современных методов SSL:**
1. Методы, основанные на metric learning;
2. Методы, основанные на self-distillation;
3. Методы, основанные на [каноническом корреляционном анализе](https://ru.wikipedia.org/wiki/%D0%9A%D0%B0%D0%BD%D0%BE%D0%BD%D0%B8%D1%87%D0%B5%D1%81%D0%BA%D0%B8%D0%B9_%D0%BA%D0%BE%D1%80%D1%80%D0%B5%D0%BB%D1%8F%D1%86%D0%B8%D0%BE%D0%BD%D0%BD%D1%8B%D0%B9_%D0%B0%D0%BD%D0%B0%D0%BB%D0%B8%D0%B7);
4. Методы, основанные на задачах восстановления информации.

## Masked Autoencoders: A PyTorch Implementation

**Masking** Following ViT, an image is divided into regular non-overlapping patches. Then a subset of  
patches is sampled and the remaining ones are masked.

**MAE encoder** The encoder is a ViT but applied only on visible, unmasked patches. Thus the encoder only operates on a small subset (~25%) of the full et. Masked patches are removed, no mask tokens are used. This allows is to train very large encoders with only a fraction of compute and memory. The full set is handled by a lightweight decoder.

**MAE decoder** The input to the MAE decoder is the full set of tokens consisting of (i) encoded visible patches, and (ii) mask tokens. Each mask token is a shared, learned vector that indicates the presence of a missing patch to be predicted. Positional embeddings are added to all tokens in this full set; without this, mask tokens would have no information about their location in the image. The decoder has another series of Transformer blocks. The MAE decoder is only used during pre-training to perform the image reconstruction task. Therefore, the decoder architecture can be flexibly designed in a manner that is independent of the encoder design.

**Reconstruction Target** MAE reconstructs the input by predicting the pixel values for each masked patch. Each element in the decoder’s output is a vector of pixel values representing a patch. The last layer of the decoder is a linear projection whose number of output channels equals the number of pixel values in a patch. The decoder’s output is reshaped to form a reconstructed image. The loss function computes the mean squared error (MSE) between the reconstructed and original images in the pixel space. Loss is computed only on masked patches, similar to BERT.

# MoE (mixture of experts) models

## What's MoE?
 In the context of transformer models, a MoE consists of two main elements:
- **Sparse MoE layers** are used instead of dense feed-forward network (FFN) layers. MoE layers have a certain number of “experts” (e.g. 8), where each expert is a neural network. In practice, the experts are FFNs, but they can also be more complex networks or even a MoE itself, leading to hierarchical MoEs!
- A **gate network or router**, that determines which tokens are sent to which expert. For example, in the image below, the token “More” is sent to the second expert, and the token "Parameters” is sent to the first network. As we’ll explore later, we can send a token to more than one expert. How to route a token to an expert is one of the big decisions when working with MoEs - the router is composed of learned parameters and is pretrained at the same time as the rest of the network.

Although MoEs provide benefits like efficient pretraining and faster inference compared to dense models, they also come with challenges:

- **Training:** MoEs enable significantly more compute-efficient pretraining, but they’ve historically struggled to generalize during fine-tuning, leading to overfitting.
- **Inference:** Although a MoE might have many parameters, only some of them are used during inference. This leads to much faster inference compared to a dense model with the same number of parameters. However, all parameters need to be loaded in RAM, so memory requirements are high. For example, given a MoE like Mixtral 8x7B, we’ll need to have enough VRAM to hold a dense 47B parameter model. Why 47B parameters and not 8 x 7B = 56B? That’s because in MoE models, only the FFN layers are treated as individual experts, and the rest of the model parameters are shared. At the same time, assuming just two experts are being used per token, the inference speed (FLOPs) is like using a 12B model (as opposed to a 14B model), because it computes 2x7B matrix multiplications, but with some layers shared (more on this soon).

## What's Sparsity?

Let’s dive deeper into Shazeer's exploration of MoEs for translation. The idea of conditional computation (parts of the network are active on a per-example basis) allows one to scale the size of the model without increasing the computation, and hence, this led to thousands of experts being used in each MoE layer.

This setup introduces some challenges. For example, although large batch sizes are usually better for performance, batch sizes in MOEs are effectively reduced as data flows through the active experts. For example, if our batched input consists of 10 tokens, **five tokens might end in one expert, and the other five tokens might end in five different experts, leading to uneven batch sizes and underutilization**.

How can we solve this? A learned gating network (G) decides which experts (E) to send a part of the input:
![[Pasted image 20240515163618.png]]

In this setup, all experts are run for all inputs - it’s a weighted multiplication. But, what happens if G is 0? If that’s the case, there’s no need to compute the respective expert operations and hence we save compute. What’s a typical gating function? In the most traditional setup, we just use a simple network with a softmax function. The network will learn which expert to send the input.
![[Pasted image 20240515163627.png]]


Shazeer’s work also explored other gating mechanisms, such as Noisy Top-k Gating. This gating approach introduces some (tunable) noise and then keeps the top k values. That is:
![[Pasted image 20240515163725.png]]
This sparsity introduces some interesting properties. By using a low enough k (e.g. one or two), we can train and run inference much faster than if many experts were activated. Why not just select the top expert? The initial conjecture was that routing to more than one expert was needed to have the gate learn how to route to different experts, so at least two experts had to be picked

Why do we add noise? That’s for load balancing!

## Load balancing with MoEs

As discussed before, if all our tokens are sent to just a few popular experts, that will make training inefficient. In a normal MoE training, the gating network converges to mostly activate the same few experts. This self-reinforces as favored experts are trained quicker and hence selected more. To mitigate this, an **auxiliary loss** is added to encourage giving all experts equal importance. This loss ensures that all experts receive a roughly equal number of training examples.

ransformers are a very clear case that scaling up the number of parameters improves the performance, so it’s not surprising that Google explored this with [GShard](https://arxiv.org/abs/2006.16668), which explores scaling up transformers beyond 600 billion parameters.

GShard replaces every other FFN layer with an MoE layer using top-2 gating in both the encoder and the decoder. The next image shows how this looks like for the encoder part. This setup is quite beneficial for large-scale computing: when we scale to multiple devices, the MoE layer is shared across devices while all the other layers are replicated
![[Pasted image 20240515164158.png]]
- **Random routing**: in a top-2 setup, we always pick the top expert, but the second expert is picked with probability proportional to its weight.
- **Expert capacity**: we can set a threshold of how many tokens can be processed by one expert. If both experts are at capacity, the token is considered overflowed, and it’s sent to the next layer via residual connections (or dropped entirely in other projects)

## Switch transformers

The Switch Transformers paper proposes a Switch Transformer layer that receives two inputs (two different tokens) and has four experts.
Contrary to the initial idea of using at least two experts, Switch Transformers uses a simplified single-expert strategy. The effects of this approach are:
- The router computation is reduced
- The batch size of each expert can be at least halved
- Communication costs are reduced
- Quality is preserved

Switch Transformers also explores the concept of expert capacity: **expert capacity = (tokens_per_batch/number_of_experts) * capacity_factor**

Switch Transformer authors also revisit and simplify the load balancing loss mentioned in the sections.** For each Switch layer, the auxiliary loss is added to the total model loss during training. This loss encourages uniform routing and can be weighted using a hyperparameter.**

Switch Transformers uses an encoder-decoder setup in which they did a MoE counterpart of T5. The [GLaM](https://arxiv.org/abs/2112.06905) paper explores pushing up the scale of these models by training a model matching GPT-3 quality using 1/3 of the energy (yes, thanks to the lower amount of computing needed to train a MoE, they can reduce the carbon footprint by up to an order of magnitude). The authors focused on decoder-only models and few-shot and one-shot evaluation rather than fine-tuning. They used Top-2 routing and much larger capacity factors. In addition, they explored the capacity factor as a metric one can change during training and evaluation depending on how much computing one wants to use.

## Stabilizing training with router Z-loss
The balancing loss previously discussed can lead to instability issues. We can use many methods to stabilize sparse models at the expense of quality. For example, introducing dropout improves stability but leads to loss of model quality. On the other hand, adding more multiplicative components improves quality but decreases stability.

Router z-loss, introduced in [ST-MoE](https://arxiv.org/abs/2202.08906), significantly improves training stability without quality degradation by penalizing large logits entering the gating network. Since this loss encourages absolute magnitude of values to be smaller, roundoff errors are reduced, which can be quite impactful for exponential functions such as the gating. We recommend reviewing the paper for details.

## What does an expert learn?

The ST-MoE authors observed that encoder experts specialize in a group of tokens or shallow concepts. For example, we might end with a punctuation expert, a proper noun expert, etc. On the other hand, the decoder experts have less specialization. The authors also trained in a multilingual setup. Although one could imagine each expert specializing in a language, the opposite happens: due to token routing and load balancing, there is no single expert specialized in any given language.
## How does scaling the number of experts impact pretraining?

More experts lead to improved sample efficiency and faster speedup, but these are diminishing gains (especially after 256 or 512), and more VRAM will be needed for inference. The properties studied in Switch Transformers at large scale were consistent at small scale, even with 2, 4, or 8 experts per layer.
## Fine-tuning MoEs

The overfitting dynamics are very different between dense and sparse models. Sparse models are more prone to overfitting, so we can explore higher regularization (e.g. dropout) within the experts themselves (e.g. we can have one dropout rate for the dense layers and another, higher, dropout for the sparse layers).

One question is whether to use the auxiliary loss for fine-tuning. The ST-MoE authors experimented with turning off the auxiliary loss, and the quality was not significantly impacted, even when up to 11% of the tokens were dropped. Token dropping might be a form of regularization that helps prevent overfitting.

Switch Transformers observed that at a fixed pretrain perplexity, the sparse model does worse than the dense counterpart in downstream tasks, especially on reasoning-heavy tasks such as SuperGLUE. On the other hand, for knowledge-heavy tasks such as TriviaQA, the sparse model performs disproportionately well. The authors also observed that a fewer number of experts helped at fine-tuning. Another observation that confirmed the generalization issue is that the model did worse in smaller tasks but did well in larger tasks.

One could experiment with freezing all non-expert weights. That is, we'll only update the MoE layers. This leads to a huge performance drop. We could try the opposite: freezing only the parameters in MoE layers, which worked almost as well as updating all parameters. This can help speed up and reduce memory for fine-tuning. This can be somewhat counter-intuitive as 80% of the parameters are in the MoE layers (in the ST-MoE project). Their hypothesis for that architecture is that, as expert layers only occur every 1/4 layers, and each token sees at most two experts per layer, updating the MoE parameters affects much fewer layers than updating other parameters.

One last part to consider when fine-tuning sparse MoEs is that they have different fine-tuning hyperparameter setups - e.g., sparse models tend to benefit more from smaller batch sizes and higher learning rates

MoEs benefit more from a higher number of tasks. Unlike the previous discussion suggesting to turn off the auxiliary loss function, the loss actually prevents overfitting.

## When to use Sparse MoEs vs Dense models?

Experts are useful for high throughput scenarios with many machines. Given a fixed compute budget for pretraining, a sparse model will be more optimal. For low throughput scenarios with little VRAM, a dense model will be better.

**Note:** one cannot directly compare the number of parameters between sparse and dense models, as both represent significantly different things.

## Making MoEs go brrrrrr

###  Parallelism
Let’s do a brief review of parallelism:

- **Data parallelism:** the same weights are replicated across all cores, and the data is partitioned across cores.
- **Model parallelism:** the model is partitioned across cores, and the data is replicated across cores.
- **Model and data parallelism:** we can partition the model and the data across cores. Note that different cores process different batches of data.
- **Expert parallelism**: experts are placed on different workers. If combined with data parallelism, each core has a different expert and the data is partitioned across all cores

With expert parallelism, experts are placed on different workers, and each worker takes a different batch of training samples. For non-MoE layers, expert parallelism behaves the same as data parallelism. For MoE layers, tokens in the sequence are sent to workers where the desired experts reside.

### Capacity Factor and communication costs
Increasing the capacity factor (CF) increases the quality but increases communication costs and memory of activations. If all-to-all communications are slow, using a smaller capacity factor is better. A good starting point is using top-2 routing with 1.25 capacity factor and having one expert per core. During evaluation, the capacity factor can be changed to reduce compute.

### Serving techniques

A big downside of MoEs is the large number of parameters. For local use cases, one might want to use a smaller model. Let's quickly discuss a few techniques that can help with serving:

- The Switch Transformers authors did early distillation experiments. By distilling a MoE back to its dense counterpart, they could keep 30-40% of the sparsity gains. Distillation, hence, provides the benefits of faster pretaining and using a smaller model in production.
- Recent approaches modify the routing to route full sentences or tasks to an expert, permitting extracting sub-networks for serving.
- Aggregation of Experts (MoE): this technique merges the weights of the experts, hence reducing the number of parameters at inference time.

### More on efficient training



## PyTorch implementation

### Understanding attention intuition

![[Pasted image 20240516080205.png]]To ensure the integrity of the autoregressive language generation process, particularly in a decoder-only model, the code implements masking. This masking technique is crucial as it obscures any information following the current token's position, thereby directing the model's attention to only the preceding parts of the sequence. Such an attention mechanism is known as causal self-attention.

Let's see how single head performs computation:
```python
torch.manual_seed(1337)
B,T,C = 4,8,32 # batch, time, channels
x = torch.randn(B,T,C)

# let's see a single Head perform self-attention
head_size = 16
key = nn.Linear(C, head_size, bias=False)
query = nn.Linear(C, head_size, bias=False)
value = nn.Linear(C, head_size, bias=False)
k = key(x)   # (B, T, 16)
q = query(x) # (B, T, 16)
wei =  q @ k.transpose(-2, -1) C**-0.5 # (B, T, 16) @ (B, 16, T) ---> (B, T, T)

tril = torch.tril(torch.ones(T, T))
#wei = torch.zeros((T,T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1) #B,T,T
wei = nn.Dropout(wei)

v = value(x) #B,T,H
out = wei @ v # (B,T,T) @ (B,T,H) -> (B,T,H)
out.shape
```
Multi-head self attention applies multiple attention heads in parallel, each focusing on a separate section of the channel (the embedding dimension). Multi-head self attention essentially improves the learning process and improves efficiency of model training due to the inherently parallel implementation
To make this multi-headed we create a Module of heads and apply them in parallel:
```
heads= nn.ModuleList([Head(head_size) for _ in range(num_heads)])
out = torch.cat([h(x) for h in self.heads],dim=-1)
```

### Creating an expert module

In the Sparse Mixture of Experts (MoE) architecture, the self-attention mechanism within each transformer block remains unchanged. However, a notable alteration occurs in the structure of each block: **the standard feed-forward neural network is replaced with several sparsely activated feed-forward networks, known as experts.**

"Sparse activation" refers to the process where each token in the sequence is routed to only a limited number of these experts – typically one or two – out of the total pool available.
```python
class Expert(nn.Module):
    """ An MLP is a simple linear layer followed by a non-linearity i.e. each Expert """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)
```
### Top-k gating  intuition

![[Pasted image 20240516081259.png]]

```python
#Understanding how gating works
num_experts = 4
top_k=2
n_embed=32

mh_output = torch.randn(2, 4, n_embed)

topkgate_linear = nn.Linear(n_embed, num_experts) # nn.Linear(32, 4)

logits = topkgate_linear(mh_output)
top_k_logits, top_k_indices = logits.topk(top_k, dim=-1)  # Get top-k experts

zeros = torch.full_like(logits, float('-inf')) 
sparse_logits = zeros.scatter(-1, top_k_indices, top_k_logits)

gating_output= F.softmax(sparse_logits, dim=-1)
gating_output

```
This idea can be extended to a noisy top-k gating for load balancing

Essentially, you don't want all the tokens to be sent to the same set of 'favored' experts. You want a fine balance of exploitation and exploration. For this purpose, to load balance, it is helpful to add standard normal noise to the logits from the gating linear layer. This makes training more efficient![[Pasted image 20240516081829.png]]

```
class NoisyTopkRouter(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(NoisyTopkRouter, self).__init__()
        self.top_k = top_k

        self.topkroute_linear = nn.Linear(n_embed, num_experts)
        self.noise_linear =nn.Linear(n_embed, num_experts)

    
    def forward(self, mh_output):
        logits = self.topkroute_linear(mh_output)
        noise_logits = self.noise_linear(mh_output)

        #Adding scaled unit gaussian noise to the logits
        noise = torch.randn_like(logits)*F.softplus(noise_logits)
        noisy_logits = logits + noise

        top_k_logits, indices = noisy_logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(noisy_logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, top_k_logits)
        router_output = F.softmax(sparse_logits, dim=-1)
        return router_output, indices
```

###  Creating a sparse Mixture of Experts module

The primary aspect of this process involves the gating network's output. After acquiring these results, the top k values are selectively multiplied with the outputs from the corresponding top-k experts for a given token. This selective multiplication forms a weighted sum, which constitutes the SparseMoe block's output

**The critical and challenging part of this process is to avoid unnecessary multiplications. It's essential to conduct forward passes only for the top_k experts and then compute this weighted sum.**

```
class SparseMoE(nn.Module):
    def __init__(self, n_embed, num_experts, top_k):
        super(SparseMoE, self).__init__()
        self.router = NoisyTopkRouter(n_embed, num_experts, top_k)
        self.experts = nn.ModuleList([Expert(n_embed) for _ in range(num_experts)])
        self.top_k = top_k

    def forward(self, x):
        gating_output, indices = self.router(x)
        final_output = torch.zeros_like(x)

        # Reshape inputs for batch processing
        flat_x = x.view(-1, x.size(-1))
        flat_gating_output = gating_output.view(-1, gating_output.size(-1))

        # Process each expert in parallel
        for i, expert in enumerate(self.experts):
            # Create a mask for the inputs where the current expert is in top-k
            expert_mask = (indices == i).any(dim=-1)
            flat_mask = expert_mask.view(-1)

            if flat_mask.any():
                expert_input = flat_x[flat_mask]
                expert_output = expert(expert_input)

                # Extract and apply gating scores
                gating_scores = flat_gating_output[flat_mask, i].unsqueeze(1)
                weighted_output = expert_output * gating_scores

                # Update final output additively by indexing and adding
                final_output[expert_mask] += weighted_output.squeeze(1)

        return final_output
```

```python

import torch
import torch.nn as nn

#Let's test this out
num_experts = 8
top_k = 2
n_embd = 16
dropout=0.1

mh_output = torch.randn(4, 8, n_embd)  # Example multi-head attention output
sparse_moe = SparseMoE(n_embd, num_experts, top_k)
final_output = sparse_moe(mh_output)
print("Shape of the final output:", final_output.shape)

Shape of the final output: torch.Size([4, 8, 16])
```
### Initialization details

Initialization is important for efficient training of deep neural nets. Kaiming He initialization is used here because of presence of ReLU activations in the experts. Feel free to experiment with Glorot initialization which is more commonly used in transformers.

```
def kaiming_init_weights(m):
    if isinstance (m, (nn.Linear)): 
        init.kaiming_normal_(m.weight)

model = SparseMoELanguageModel()
model.apply(kaiming_init_weights)
```

### Altogether
```
class SparseMoELanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head, num_experts=num_experts,top_k=top_k) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed) # final layer norm
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

```
### Training

```python

#Using MLFlow
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
#mlflow.set_experiment("makeMoE")
with mlflow.start_run():
    #If you use mlflow.autolog() this will be automatically logged. I chose to explicitly log here for completeness
    params = {"batch_size": batch_size , "block_size" : block_size, "max_iters": max_iters, "eval_interval": eval_interval,
              "learning_rate": learning_rate, "device": device, "eval_iters": eval_iters, "dropout" : dropout, "num_experts": num_experts, "top_k": top_k }
    mlflow.log_params(params)
    for iter in range(max_iters):

        # every once in a while evaluate the loss on train and val sets
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss()
            print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            metrics = {"train_loss": losses['train'], "val_loss": losses['val']}
            mlflow.log_metrics(metrics, step=iter)


        # sample a batch of data
        xb, yb = get_batch('train')

        # evaluate the loss
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
```

# Autoencoders
**Autoencoders** are a specific type of feedforward neural networks where the input is the same as the output. They compress the input into a lower-dimensional latent representation and then reconstruct the output from this representation.
![[0_32-VK98ppmMz-Ogj.webp]]
Autoencoders are mainly used for **dimensionality reduction** (or compression) with a couple of important properties:

- Autoencoders are only able to meaningfully compress data similar to what they have been trained on. Since they learn features specific for the given training data, they are different than a standard data compression algorithm like gzip. So we can’t expect an autoencoder trained on handwritten digits to compress landscape photos.
- The output of the autoencoder will not be exactly the same as the input, it will be a close but degraded representation. If you want lossless compression they are not the way to go.
- To train an autoencoder we don’t need to do anything fancy, just throw the raw input data at it. Autoencoders are considered an unsupervised learning technique since they don’t need explicit labels to train on. But to be more precise they are self-supervised because they generate their own labels from the training data.

There are **4 hyperparameters** that we need to set before training an autoencoder:
- Latent vector dimensions
- Loss 
- number of layers
- nodes per layer

## Variants of AEs

Convolutional autoencoders leverage convolutional layers to excel in image-related tasks, capturing spatial relationships effectively. Sparse autoencoders introduce sparsity constraints on the latent space activations, aiding feature learning and dimensionality reduction. Denoising autoencoders tackle noise by training on corrupted versions of input data, leading to robust feature extraction. Contractive autoencoders include penalty terms in the loss function to enhance stability and reduce sensitivity to input variations. Stacked autoencoders combine multiple layers of autoencoders to create deep architectures for hierarchical feature learning. Finally, variational autoencoders (VAEs) inject probabilistic elements into the latent space, enabling data generation and intricate feature disentanglement. Now, we will go over a few details of Sparse AE and Denoising AE.

### Sparse AE
A sparse autoencoder is simply an autoencoder whose training criterion involves a sparsity penalty. In most cases, we would construct our loss function by penalizing activations of hidden layers so that only a few nodes are encouraged to activate when a single sample is fed into the network.

So, in sparse autoencoder we add L1 penalty to the loss to learn sparse feature representations. L1 regularization adds “absolute value of magnitude” of coefficients as penalty term. Although L1 and L2 can both be used as regularization term, the key difference between them is that L1 regularization tends to shrink the penalty coefficient to zero while L2 regularization would move coefficients towards zero but they will never reach. Thus L1 regularization is often used as a method of feature extraction. Hence the loss function will be:![[0_FfUAFQxja8EUqzpu.webp]]
### Denoising AE
### Convolutional AE
### Variational AE

A variational autoencoder (VAE) converts the input data to a variational representation vector (as the name suggests), where the elements of this vector represent different attributes about the input data distribution. This _probabilistic_ property of the VAE makes it a generative model.

**The latent representation in VAE is composed of a probability distribution (_μ,_ σ) that best defines our input data**

in order to be able to use the decoder of our autoencoder for generative purpose, we have to be sure that the latent space is regular enough. One possible solution to obtain such regularity is to introduce explicit regularisation during the training process. Thus, as we briefly mentioned in the introduction of this post, **a variational autoencoder can be defined as being an autoencoder whose training is regularised to avoid overfitting and ensure that the latent space has good properties that enable generative process.**

In order to introduce some regularisation of the latent space, we proceed to a slight modification of the encoding-decoding process: **instead of encoding an input as a single point, we encode it as a distribution over the latent space**.

The model is then trained as follows:

- first, the input is encoded as distribution over the latent space
- second, a point from the latent space is sampled from that distribution
- third, the sampled point is decoded and the reconstruction error can be computed
- finally, the reconstruction error is backpropagated through the network

In practice, the encoded distributions are chosen to be normal so that the encoder can be trained to return the mean and the covariance matrix that describe these Gaussians.

**The loss function that is minimised when training a VAE is composed of a “reconstruction term” (on the final layer), that tends to make the encoding-decoding scheme as performant as possible, and a “regularisation term” (on the latent layer), that tends to regularise the organisation of the latent space by making the distributions returned by the encoder close to a standard normal distribution. That regularisation term is expressed as the [Kulback-Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) between the returned distribution and a standard Gaussian**
![[1_Q5dogodt3wzKKktE0v3dMQ@2x.webp]]
####  Intuitions about the regularisation

The regularity that is expected from the latent space in order to make generative process possible can be expressed through two main properties: **continuity** (two close points in the latent space should not give two completely different contents once decoded) and **completeness** (for a chosen distribution, a point sampled from the latent space should give “meaningful” content once decoded)

The only fact that VAEs encode inputs as distributions instead of simple points is not sufficient to ensure continuity and completeness. Without a well defined regularisation term, the model can learn, in order to minimise its reconstruction error, **to “ignore” the fact that distributions are returned and behave almost like classic autoencoders** (leading to overfitting). To do so, the encoder can either return distributions with tiny variances (that would tend to be punctual distributions) or return distributions with very different means (that would then be really far apart from each other in the latent space). In both cases, distributions are used the wrong way (cancelling the expected benefit) and continuity and/or completeness are not satisfied.

So, in order to avoid these effects **we have to regularise both the covariance matrix and the mean of the distributions returned by the encoder**. In practice, this regularisation is done by enforcing distributions to be close to a standard normal distribution (centred and reduced). This way, we require the covariance matrices to be close to the identity, preventing punctual distributions, and the mean to be close to 0, preventing encoded distributions to be too far apart from each others.

we can observe that continuity and completeness obtained with regularisation **tend to create a “gradient” over the information encoded in the latent space**. For example, a point of the latent space that would be halfway between the means of two encoded distributions coming from different training data should be decoded in something that is somewhere between the data that gave the first distribution and the data that gave the second distribution as it may be sampled by the autoencoder in both cases.

#### Mathematical details

Let’s begin by defining a probabilistic graphical model to describe our data. We denote by x the variable that represents our data and assume that x is generated from a latent variable z.

Thus, for each data point, the following two steps generative process is assumed:

- first, a latent representation z is sampled from the prior distribution p(z)
- second, the data x is sampled from the conditional likelihood distribution p(x|z)

The “probabilistic decoder” is naturally defined by **p(x|z), that describes the distribution of the decoded variable given the encoded one**, whereas the “probabilistic encoder” is defined by **p(z|x), that describes the distribution of the encoded variable given the decoded one**.

At this point, we can already notice that the regularisation of the latent space that we lacked in simple autoencoders naturally appears here in the definition of the data generation process: encoded representations z in the latent space are indeed assumed to follow the prior distribution p(z). Otherwise, we can also remind the the well-known Bayes theorem that makes the link between the prior p(z), the likelihood p(x|z), and the posterior p(z|x)

![[Pasted image 20240521011548.png]]

Let’s now make the assumption that p(z) is a standard Gaussian distribution and that p(x|z) is a Gaussian distribution whose mean is defined by a deterministic function f of the variable of z and whose covariance matrix has the form of a positive constant c that multiplies the identity matrix.

he function f is assumed to belong to a family of functions denoted F that is left unspecified for the moment and that will be chosen later. Thus, we have

![[Pasted image 20240521012150.png]]
Let’s consider, for now, that f is well defined and fixed. In theory, as we know p(z) and p(x|z), we can use the Bayes theorem to compute p(z|x): this is a classical [Bayesian inference problem](https://towardsdatascience.com/bayesian-inference-problem-mcmc-and-variational-inference-25a8aa9bce29).

In statistics, **variational inference (VI) is a technique to approximate complex distributions**. The idea is to set a parametrised family of distribution (for example the family of Gaussians, whose parameters are the mean and the covariance) and to look for the best approximation of our target distribution among this family. The best element in the family is one that minimise a given approximation error measurement (most of the time the Kullback-Leibler divergence between approximation and target) and is found by gradient descent over the parameters that describe the family.

Here we are going to approximate p(z|x) by a Gaussian distribution q_x(z) whose mean and covariance are defined by two functions, g and h, of the parameter x. These two functions are supposed to belong, respectively, to the families of functions G and H that will be specified later but that are supposed to be parametrised. Thus we can denote:

![[Pasted image 20240521012438.png]]
So, we have defined this way a family of candidates for variational inference and need now to find the best approximation among this family by optimising the functions g and h to minimise the Kullback-Leibler divergence between the approximation and the target p(z|x).

![[Pasted image 20240521012622.png]]
Up to know, we have assumed the function f known and fixed and we have showed that, under such assumptions, we can approximate the posterior p(z|x) using variational inference technique. However, in practice this function f, that defines the decoder, is not known and also need to be chosen.

**For a given input x, we want to maximise the probability to have x̂ = x when we sample z from the distribution q*_x(z) and then sample x̂ from the distribution p(x|z).** Thus, we are looking for the optimal f* such that

![[Pasted image 20240521013449.png]]
The higher c is the more we assume a high variance around f(z) for the probabilistic decoder in our model and, so, the more we favour the regularisation term over the reconstruction term (and the opposite stands if c is low).
#### Bringing neural networks into the model

 As we can’t easily optimise over the entire space of functions, we constrain the optimisation domain and decide to express f, g and h as neural networks. Thus, F, G and H correspond respectively to the families of functions defined by the networks architectures and the optimisation is done over the parameters of these networks.

In practice, g and h are not defined by two completely independent networks but share a part of their architecture and their weights so that we have.
![[Pasted image 20240521014127.png]]
As it defines the covariance matrix of q_x(z), h(x) is supposed to be a square matrix. However, in order to simplify the computation and reduce the number of parameters, we make the additional assumption that our approximation of p(z|x), q_x(z), is a multidimensional Gaussian distribution with diagonal covariance matrix (variables independence assumption). With this assumption, h(x) is simply the vector of the diagonal elements of the covariance matrix and has then the same size as g(x). However, we reduce this way the family of distributions we consider for variational inference and, so, the approximation of p(z|x) obtained can be less accurate.


Our model assumes for p(x|z) a Gaussian with fixed covariance. The function f of the variable z defining the mean of that Gaussian is modelled by a neural network and can be represented as follows

![[Pasted image 20240521015448.png]]
The overall architecture is then obtained by concatenating the encoder and the decoder parts. However we still need to be very careful about the way we sample from the distribution returned by the encoder during the training. The sampling process has to be expressed in a way that allows the error to be backpropagated through the network.

 A simple trick, called **reparametrisation trick**, is used to make the gradient descent possible despite the random sampling that occurs halfway of the architecture and consists in using the fact that if z is a random variable following a Gaussian distribution with mean g(x) and with covariance H(x)=h(x).h^t(x) then it can be expressed as:

![[Pasted image 20240521015629.png]]
![[Pasted image 20240521015744.png]]
Finally, the objective function of the variational autoencoder architecture obtained this way is given by the last equation of the previous subsection in which the theoretical expectancy is replaced by a more or less accurate Monte-Carlo approximation that consists, most of the time, into a single draw. 

So, considering this approximation and denoting C = 1/(2c), we recover the loss function derived intuitively in the previous section, composed of a reconstruction term, a regularisation term and a constant to define the relative weights of these two terms.

![[Pasted image 20240521015856.png]]
#### PyTorch code

The input dimension is 784 which is the flattened dimension of MNIST images (28×28). In the encoder, the mean (μ) and variance (σ²) vectors are our variational representation vectors (size=200). Notice that we multiply the latent variance with the epsilon (ε) parameter for reparameterization before decoding. This allows us to perform backpropagation and tackle the node stochasticity.

Also, our final encoder dimension has dimension 2 which are the μ and σ vectors. These continuous vectors define our latent space distribution that allows us to sample images in VAE.

```
class VAE(nn.Module):  
  
def __init__(self, input_dim=784, hidden_dim=400, latent_dim=200, device=device):  
super(VAE, self).__init__()  
  
# encoder  
self.encoder = nn.Sequential(  
nn.Linear(input_dim, hidden_dim),  
nn.LeakyReLU(0.2),  
nn.Linear(hidden_dim, latent_dim),  
nn.LeakyReLU(0.2)  
)  
  
# latent mean and variance  
self.mean_layer = nn.Linear(latent_dim, 2)  
self.logvar_layer = nn.Linear(latent_dim, 2)  
  
# decoder  
self.decoder = nn.Sequential(  
nn.Linear(2, latent_dim),  
nn.LeakyReLU(0.2),  
nn.Linear(latent_dim, hidden_dim),  
nn.LeakyReLU(0.2),  
nn.Linear(hidden_dim, input_dim),  
nn.Sigmoid()  
)  
  
def encode(self, x):  
x = self.encoder(x)  
mean, logvar = self.mean_layer(x), self.logvar_layer(x)  
return mean, logvar  
  
def reparameterization(self, mean, var):  
epsilon = torch.randn_like(var).to(device)  
z = mean + var*epsilon  
return z  
  
def decode(self, x):  
return self.decoder(x)  
  
def forward(self, x):  
mean, logvar = self.encode(x)  
z = self.reparameterization(mean, logvar)  
x_hat = self.decode(z)  
return x_hat, mean, log_var

```

```
model = VAE().to(device)  
optimizer = Adam(model.parameters(), lr=1e-3)
```

The loss function in VAE consists of reproduction loss and the Kullback–Leibler (KL) divergence. The KL divergence is a metric used to measure the distance between two probability distributions.

```
def loss_function(x, x_hat, mean, log_var):  
reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')  
KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())  
  
return reproduction_loss + KLD
```

Simple training: 

```
def train(model, optimizer, epochs, device):  
model.train()  
for epoch in range(epochs):  
overall_loss = 0  
for batch_idx, (x, _) in enumerate(train_loader):  
x = x.view(batch_size, x_dim).to(device)  
  
optimizer.zero_grad()  
  
x_hat, mean, log_var = model(x)  
loss = loss_function(x, x_hat, mean, log_var)  
  
overall_loss += loss.item()  
  
loss.backward()  
optimizer.step()  
  
print("\tEpoch", epoch + 1, "\tAverage Loss: ", overall_loss/(batch_idx*batch_size))  
return overall_loss  
  
train(model, optimizer, epochs=50, device=device)
```

To generate a simple image is to:
```
def generate_digit(mean, var):  
z_sample = torch.tensor([[mean, var]], dtype=torch.float).to(device)  
x_decoded = model.decode(z_sample)  
digit = x_decoded.detach().cpu().reshape(28, 28) # reshape vector to 2d array  
plt.imshow(digit, cmap='gray')  
plt.axis('off')  
plt.show()  
  
generate_digit(0.0, 1.0), generate_digit(1.0, 0.0)
```

## Limitations
1. Deterministic
2.  Computationally expensive when dealing with large and complex datasets

  While they can be used to generate reconstructions of the input data, they do not inherently learn to generate entirely new data samples from scratch. The primary objective of autoencoders is to capture the most informative and relevant features in the data and produce accurate reconstructions. The latent space learned by autoencoders does not follow any specific probability distribution. Thus the AEs are deterministic and not generative.

Sampling from the vanilla autoencoders is also difficult because its learned distribution is oddly shaped, discontinuous, and not centered at (0,0). There are areas called as holes with no points between the images of different types. Thus, if a point is randomly picked up from this empty space, it is not certain whether the desired images will get created. Autoencoder has no way to ensure that the points in the latent space are continuous in nature.

# DSSM (deep semantic similarity model) и причём тут суммаризация

Она была описана в 2013 году, в [статье от Microsoft](https://www.microsoft.com/en-us/research/publication/learning-deep-structured-semantic-models-for-web-search-using-clickthrough-data), где они предложили такую архитектуру нейронной сети, в которой для каждой сущности выделяется отдельная ветвь, заканчивающаяся семантическим представлением этой сущности — вектором эмбеддингом. Мы хотим, чтобы эти векторы обладали определенным свойством, например, если две сущности похожи друг на друга, то и соответствующие им векторы должны быть близки в некотором векторном пространстве. А если эти сущности никак не связаны, то и векторы должны быть направлены в разные стороны.

![[Pasted image 20240529164510.png]]
![[Pasted image 20240529164557.png]]

> Но, пожалуй, центральное место в этой архитектуре занимает обработка текстовой информации — описание документа, например, описание вакансии или опыта в резюме. Этот текст мы передаем на вход слою RNN, который на выходе выдает по одному вектору на каждый входной токен, далее агрегируем эти вектора с помощью простого линейного attention-слоя, затем конкатенируем все полученные признаки и прогоняем еще через пару слоев нейронной сети. Таким образом на выходе получаем тот самый желанный вектор эмбеддинг или семантическое представление вакансии. Аналогичная архитектура у нас также есть и для резюме, она будет отличаться просто набором входных признаков, а на выходе мы тоже получаем вектор представления для резюме.
> Получив два вектора по вакансии и резюме, мы можем найти расстояние между ними и запустить процесс обучения. Мы считаем, что вакансии и резюме должны быть похожи друг на друга, если между ними был какой-то позитивный сигнал. Например, был отклик на эту вакансию от кандидата или было приглашение на интервью. Негативные примеры можем насэмплировать либо случайно, либо взяв те вакансии, на которые человек просто посмотрел, но ничего больше не сделал.
> На вход слою RNN мы даем не весь текст вакансии или резюме, а только первые 300 токенов.


 Серым выделено то, что не поместилось в первые 300 токенов, и мы видим, что первый абзац занял очень много места, а технологии не влезли вовсе. Отсюда у нас возникла идея построить такую модель, которая будет автоматически выделять самые важные части из описания документа. Тогда мы получим более полное и правильное семантическое представление этого документа.

Так у нас родилась задача суммаризации — построения краткого содержания текста.


**Существуют различные подходы к решению задачи суммаризации — они делятся на экстрактивные и абстрактивные. Экстрактивные подходы — это когда для суммаризации используются кусочки оригинального текста в том виде, как они были написаны изначально. Абстрактивная суммаризация — это генеративная модель, которая создает абсолютно новый текст, где могут содержаться слова, которые даже не встречались в оригинальном тексте.**

Следующее деление подходов к решению задачи суммаризации делится на supervised- и unsupervised-learning. Простейший пример подхода unsupervised-learning — это алгоритм [TextRank](https://towardsdatascience.com/text-summarization-with-nlp-textrank-vs-seq2seq-vs-bart-474943efeb09), он работает следующим образом: мы берем текст, разбиваем его на предложения, векторизуем каждое, а затем строим граф, где вершинами станут наши предложения, а ребра мы взвесим расстояниями между соответствующими предложениями.

Каким образом векторизовать предложение или как посчитать эти расстояния — рождается масса вариаций этого алгоритма. Но потом мы запускаем на этом графе алгоритм PageRank, который выдает нам скоры для каждой вершины, и эти скоры мы уже можем проинтерпретировать как важность соответствующих предложений. После этого мы берем самые важные предложения и строим из них суммаризацию.

Наш выбор — это supervised-learning. Для этого нам необходима выборка из текстов с разметкой, например, по предложениям. В нем будет указано, какое предложение релевантное и его нужно оставить в саммари, а какое — нет.

Одна проблема — построить такой сэмпл вручную практически нереально. Чтобы человеку вручную разметить одну вакансию, потребуется очень много времени. Да и в принципе не до конца понятно, что именно ему нужно размечать, потому что задача стоит не слишком конкретно — отобрать такие предложения, выбрав из которых 300 токенов и передав их на вход нашей модели DSSM, мы получим более правильное и полноценное семантическое представление документа. Но как определить самые важные предложения? Тут у нас родилась следующая идея. Почему бы не спросить у самой модели DSSM, что ей важно, а что нет.

Вернемся к нашей архитектуре DSSM и подробнее разберем слой линейного аттеншна, который я упоминал ранее. На самом деле этот слой достаточно простой. Во-первых, на вход в слой RNN у нас поступает 300 токенов, а на выходе мы получаем по вектору на каждый входной токен. Грубо говоря, 300 векторов размерности 256.

Далее у нас есть простой линейный слой, который перемножает каждый этот вектор, а на выходе дает нам одно число — получается один вектор размерности 300. Теперь этот вектор пропускаем через softmax, который его нормирует, и в итоге получаем веса. Эти веса используются для того, чтобы посчитать взвешенную сумму выходов из RNN и таким образом получить результирующий вектор размерности 256.

За счет этой схемы модель учится сама, какому токену придать больше веса, какому меньше. Проще говоря, от какого токена взять больше информации в этой результирующей сумме, а какой вектор можно практически полностью занулить.

Частый трюк в машинном обучении — аппроксимация, замена чего-то сложного и большого на более простую модель, которая работает с погрешностью. Здесь мы тоже можем применить этот трюк и обучить некоторую простую модель, которая бы предсказывала разметку аттаншена нашей расширенной модели. Таким образом мы бы и решили задачу суммаризации, то есть бы выделяли те части вакансии, которые содержат самую релевантную информацию.

![[Pasted image 20240529165420.png]]



Работу логистической регрессии мы можем использовать в качестве модели суммаризации следующим образом: берем текст нашего документа, разбиваем его на предложения, строим сэмпл и далее. Каждое предложение прогоняем через обученную логистическую регрессию, которая, как известно, выдает нам некоторое значение от нуля до единицы. Это значение мы можем проинтерпретировать как релевантность каждого предложения. Далее сортируем эти предложения, исходя из этого скора, и отбираем из них только самые важные, пока не наберется 300 токенов. Всё остальное выкидываем. Таким образом, у нас остается саммари из самых важных предложений, которые в итоге мы еще сортируем в том порядке, как они шли изначально

![[Pasted image 20240529165620.png]]


# MAB and CAB

## Глоссарий 
```
**Вариант (стратегия, action, arm)** - аналог группы в A/B-тесте. Какое-то изменение, которое вы хотите сравнить с текущей реализацией или чем-либо еще.

**Награда (целевая метрика, reward)** - значение метрики, которое получено от использования того или иного варианта.

**Убыток (regret)** - потери в награде от использования того или иного варианта.

**Среда** - совокупность некоторых условий и отношений, которая формирует определенные закономерности поведения субъектов среды (например: рынок маркетплейсов).

**Контекст (факторы)** - определенные параметры, характеризующие субъекта среды (пол, возраст, страна).
```

## Суть
![[Pasted image 20240531144955.png]]

**Многорукий бандит** — это алгоритм принятия решений, суть которого заключается в обеспечении баланса между исследованием и использованием какого-то из рассматриваемых вариантов решения с целью максимизации целевой метрики и минимизации убытков.

1. Классические многорукие бандиты (MAB). Это алгоритмы для выбора одного самого эффективного варианта с максимальной потенциальной наградой. Это самый распространённый тип многоруких бандитов, к нему относятся широко известные классические реализации MAB: epsilon-greedy, UCB, Thompson sampling. Алгоритмы MAB не способны улавливать контекстные особенности среды, в которой работают, — они лишь перераспределяют трафик на варианты, где среднестатистически наблюдается наибольшее среднее значение целевой метрики.
2. Контекстные бандиты (CMAB). Это алгоритмы, направленные на оптимизацию награды в зависимости от контекста среды, в которой они работают. Наиболее известные алгоритмы CMAB являются по своей сути комбинацией epsilon-greedy, UCB, Thompson sampling и линейной регрессии: linear epsilon-greedy, UCB, TS. 

У алгоритмов MAB и CMAB есть границы применимости. Сразу обозначу: они не являются полноценной заменой A/B-тестирования, у них есть определённая ниша, в которой их применение считается наиболее подходящим. **Для CMAB — это задачи, связанные с персонализацией: рекомендательные системы, оптимизация промо, динамическое ценообразование, контекстная реклама и т. д. Для MAB это задачи, где необходимо выбрать одно решение, но проведение A/B-теста затруднено.**


1. Оба типа многоруких бандитов могут оптимизировать лишь одну метрику, поэтому она должна полноценно отражать суть решаемой задачи. Например, если вы хотите повысить маржинальность продаж, нужно выбрать маржу в качестве метрики, а если цель — увеличить CTR, то очевидно, что следует выбрать в качестве метрики конверсию в клик. 
    

2. CMAB и MAB могут приспосабливаться (меняя выигрышный вариант) к изменениям условий среды, которую они исследуют, однако делают они это по-разному. Если для CMAB это вообще основа работы, то с MAB дело обстоит сложнее: часть алгоритма, ответственная за исследование, может заметить изменение лучшего варианта в определённый момент времени либо проигнорировать этот факт вследствие полного перехода в использование уже выбранного варианта. Последнее часто происходит, если на раннем этапе работы MAB среди вариантов был явный лидер. Это особенно характерно для UCB. Чтобы предотвратить подобную ситуацию и сделать алгоритм MAB чувствительным к изменениям среды, следует предусмотреть возможность переключения раундов с полным обновлением весов MAB либо использовать реализации алгоритма, в которых  предусмотрено ручное включение случайного выбора ручек для обновления знаний о вариантах-аутсайдерах в текущем раунде бандита.

Преимущества и недостатки MAB и CMAB:

Преимущества:
- гибкость исследования и возможность перераспределять трафик между худшими и лучшими вариантами для минимизации убытков; 
- возможность решить задачу, когда A/B-тест применять нецелесообразно; 
- возможность персонализированно выбирать лучший вариант (CMAB). 
Недостатки: 
- отсутствие статистических предпосылок для фиксации уровней ошибок I и II рода; 
- мощность MAB может быть ниже, чем у A/B-теста, особенно если разница между вариантами очень мала.

## Теоретическая основа MAB

Основными алгоритмами классических многоруких бандитов являются: 

- **ε-greedy** 
- **UCB (upper confidence bound)** 
- **Thompson sampling.**

Суть алгоритма **ε-greedy** очень проста: выбираем стратегию с максимальной средней наградой (средним значением метрики, которую мы оптимизируем) и иногда с определённой заранее вероятностью выбираем случайную стратегию для исследования.


**UCB**, в отличие от ε-greedy, проводит своё исследование не случайно, а на основе растущей со временем неопределённости у стратегий. В начале работы алгоритм случайно задействует все стратегии, после чего рассчитывается средняя награда каждой. Далее выбор стратегии происходит следующим образом: 
1. После каждой итерации обновляются средние награды стратегий. 
2. С течением времени чем реже выбиралась та или иная стратегия, тем больше будет у неё неопределённость. 
3. Окончательный выбор стратегии — это максимальная сумма средней награды и неопределённости среди всех стратегий. 
![[Pasted image 20240531145912.png]]
где: 
- _Qt(a)_ — среднее значение награды стратегии _a_; 
- _t_ — общее количество наблюдений;
- _Nt(a)_ — количество раз, когда была выбрана стратегия _a;_ 
- _c_ — коэффициент бустинга исследования. Чем он больше, тем больше алгоритм направлен на исследование. (Вообще этот коэффициент может быть как статичным, так и динамичным. Так как награда может меняться со временем, лучше сделать его зависимым от её значения.)

[**Thompson sampling**](https://www.google.com/url?q=https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf&sa=D&source=docs&ust=1685370309094896&usg=AOvVaw2iEcch3-Pf_IK76uvDczhS) — самый сложный алгоритм MAB. Он основан на байесовском подходе, поэтому с ним неразрывно связано два термина:
- Априорное распределение - распределение, которое выражает предположения до учета экспериментальных данных.
- Апостериорное распределение - распределение, которое получено после учёта экспериментальных данных.

У каждого варианта перед запуском бандита есть априорное распределение его награды, которое по мере поступления новых данных становится апостериорным. Сэмплирование Томпсона берет рандомные значения из этих распределений, сравнивает их и выбирает вариант с максимальным значением.

Давайте рассмотрим реализацию Thompson sampling для [биномиальной метрики](https://www.google.com/url?q=https://gdmarmerola.github.io/ts-for-bernoulli-bandit/&sa=D&source=docs&ust=1685370289544712&usg=AOvVaw0XS81AWTYsuz9aZnY5PTka). В качестве априорного распределения возьмём бета-распределение с параметрами α и β: ![[Pasted image 20240531150603.png]]
**где α_k и β_k являются по своей сути кумулятивной суммой количества удачных и неудачных исходов:**

Выбор стратегии в данном случае — максимальное значение theta, полученное из апостериорных распределений наших стратегий.

Следует отметить, что все эти классические алгоритмы можно и нужно улучшать и комбинировать по своему усмотрению. Например, можно скомбинировать UCB и ε-greedy для добавления случайного переключения в алгоритм UCB, причём можно это реализовать не просто как единоразовое случайное переключение, а как период, в течение которого алгоритм рандомно распределяет трафик, а после снова возвращается к своему первоначальному функционированию.
## Теоретическая основа CMAB
1. Определить для каждого фактора модели байесовской линейной регрессии априорное распределение: Prior(mu, std), где mu = 0, std = 1 (или любые другие значения, которые вы посчитаете подходящими). 
    
2. Определить функцию для сэмплирования Томпсона:

![BestAction(X_t, Prior, action)](https://habrastorage.org/getpro/habr/upload_files/edf/6e4/d75/edf6e4d752d10e49d7e170771cfbad6c.svg)

3. Вариант выбирается следующим образом: для каждого варианта, для каждого параметра модели рандомно сэмплируются значения _βk_ из априорного распределения и умножаются на _Xk_. Полученные значения предикта далее сравниваются между вариантами — и выбирается максимальное значение.     
4. Запустить первую итерацию контекстного бандита и собрать первый батч данных (в первой итерации выбор из априорного распределения идентичен рандомному сэмплированию): 
![arm_t = BestAction(X_t, Prior, actions)](https://habrastorage.org/getpro/habr/upload_files/986/085/1a2/9860851a24f4073caa259421584fe56e.svg)
5. На данных первого батча обучить байесовскую линейную регрессию для каждого из предоставленных вариантов. Сохранить параметры полученного апостериорного распределения (среднее и стандартное отклонения):  
![Prior_{new} = Posterior(mu, std)](https://habrastorage.org/getpro/habr/upload_files/013/560/e59/013560e59631c79b2b01dbb22fa86d45.svg)
6. Используя новое апостериорное распределение, сделать предикт для выбора лучшего варианта:
![arm_t = BestAction(Xt, Prior_{new}, actions)](https://habrastorage.org/getpro/habr/upload_files/b2e/efe/578/b2eefe57894e51eaeb8c651d9668b60e.svg)
7. Повторить пункты 5 и 6 для новых батчей данных n раз.

Контекстный бандит linear TS готов! Однако проблема в том, что он линейный, то есть будет довольно плохо оптимизировать нелинейные связи, которых на практике подавляющее большинство. Решение данной проблемы описали в статье [Deep Bayesian Bandits Showdown](https://www.google.com/url?q=https://research.google/pubs/pub46647/&sa=D&source=docs&ust=1685369012077265&usg=AOvVaw1xQGLPhuGvos4SkrtWoJfJ).

Для улучшения интерпретации нелинейных связей между таргетом и факторами необходимо реализовать многоуровневый алгоритм neural linear TS, который состоит из двух моделей: нейронной сети и линейной байесовской регрессии.

Вместо того чтобы напрямую передавать наши фичи в линейную регрессию, как мы это делали в случае с линейным бандитом, мы сначала обучим на этих фичах нейронную сеть и потом возьмём аутпуты из её последнего скрытого слоя  в качестве новых фич, которые в свою очередь будут использованы для обучения линейной регрессии. Эти новые фичи, по сути, являются репрезентацией наших исходных факторов, которые призваны облегчить линейной модели оптимизацию нелинейных зависимостей. Данный алгоритм действий представлен в виде схемы на рисунке далее.
![[Pasted image 20240531151555.png]]


## Практическое применение и валидация CMAB

Так как мы не можем знать, какое априорное распределение у β-параметров нашей модели, возьмём нормальное распределение со средним 0 и стандартным отклонением 1. 

Реализуем функцию для определения априорного распределения для параметров модели. В неё нужно передать количество вариантов и количество фич модели:

```
def get_priors(arms, n_features):    
	posteriors = {}    
	for arm in arms:
		m = np.zeros(n_features)
		s = np.ones(n_features)
		posteriors[arm] = [m, s]
	return posteriors
```

Реализуем функцию для сэмплирования Томпсона (функцию BestAction из теоретической части).
α — параметр, который влияет на exploration / exploitation tradeoff. Чем меньше α, тем меньше дисперсия у распределения β и тем больше exploitation.

```
def select_arm_thompson(posteriors, context, arms, alpha = 1):  
	samples = {}
	context = np.insert(context, 0, 1)
	for arm in arms:
		m = posteriors[arm][0][:-1]
		s = posteriors[arm][1][:-1]* alpha
		w = np.random.normal(m, s)
		sample_prediction = np.dot(w, context)
		samples[arm] = sample_prediction
	max_value = max(samples.values());
	max_keys = [key for key, value in samples.items() if value == max_value]  
	return np.random.choice(max_keys)
```

Последнее, что нам нужно, — это сама модель. В PyMC есть хороший API [](https://bambinos.github.io/bambi/main/index.html)[Bambi](https://bambinos.github.io/bambi/) (BAyesian Model-Building Interface), с помощью которого можно легко построить нужную нам модель и задать априорное распределение для параметров.

(В качестве среды симуляции используем [toy problem](https://www.google.com/url?q=https://github.com/LaunchpadAI/space-bandits/blob/master/toy_problem.ipynb&sa=D&source=docs&ust=1685369501292834&usg=AOvVaw0HFgzJpgA8Bu--ILIAyGYV) из библиотеки [Space Bandits (SB)](https://www.google.com/url?q=https://github.com/LaunchpadAI/space-bandits&sa=D&source=docs&ust=1685369517639214&usg=AOvVaw02hEdc2rm27HYrFfqsKCgL).)

## Выкат в прод

1. Провести симуляцию, как это сделали мы в toy problem, но на реальных данных. С MAB это не вызывает проблем — процесс похож на симуляцию A/B-тестирования, но с CMAB придётся немного напрячься. Необходимо построить функциональную зависимость между целевой метрикой и факторами пользователя, а также разделить пользователей по паттернам поведения. Тогда вы сможете приблизительно оценить, как будет работать CMAB на реальных данных.
2. Подготовить A/B-тест для контроля за эффективностью работы бандита. Без эксперимента не стоит запускать бандита, так как в таком случае вы просто лишитесь возможности оценить эффективность алгоритма.

# AirLLM

## Layer-wise inference

During inference, layers are executed sequentially. The output of the previous layer is the input to the next. Only one layer executes at a time.
Therefore, it is completely unnecessary to keep all layers in GPU memory. **We can load whichever layer is needed from disk when executing that layer, do all the calculations, and then completely free the memory after.**
his way, the GPU memory required per layer is only about the parameter size of one transformer layer, 1/80 of the full model, around 1.6GB.

In addition, some output caches are also stored in GPU memory, the largest being the KV cache to avoid repeated computations.

A simple calculation, for the 70B model this KV cache size is about:

2 * input_length * num_layers * num_heads * vector_dim * 4

With input length 100, this cache = 2 * 100 * 80 * 8 * 128 * 4 = 30MB GPU memory.

**According to huggingface monitoring, the entire inference process uses less than 4GB GPU memory!**
## Single layer optimization Flash-attention

**Scaling the transformer architecture is heavily bottlenecked by the self-attention mechanism, which has quadratic time and memory complexity**. Recent developments in accelerator hardware mainly focus on enhancing compute capacities and not memory and transferring data between hardware. This results in attention operation having a memory bottleneck. 

**Flash Attention** is an attention algorithm used to reduce this problem and scale transformer-based models more efficiently, enabling faster training and inference.

Standard attention mechanism uses High Bandwidth Memory (HBM) to store, read and write keys, queries and values. 

**HBM is large in memory, but slow in processing, meanwhile SRAM is smaller in memory, but faster in operations**. In the standard attention implementation, the cost of loading and writing keys, queries, and values from HBM is high. 
**It loads keys, queries, and values from HBM to GPU on-chip SRAM, performs a single step of the attention mechanism, writes it back to HBM, and repeats this for every single attention step. Instead, Flash Attention loads keys, queries, and values once, fuses the operations of the attention mechanism, and writes them back.**


![[Pasted image 20240613133141.png]]
![[Pasted image 20240614091359.png]]

### Key takeaways
The takeaway is that FlashAttention is:

- **Fast** — excerpt from the paper: “We train BERT-large (seq. length 512) 15% faster than the training speed record in MLPerf 1.1, GPT2 (seq. length 1K) 3x faster than baseline implementations from HuggingFace and Megatron-LM, and long-range arena (seq. length 1K-4K) 2.4x faster than baselines.”
- **Memory-efficient** — compared to vanilla attention, which is quadratic in sequence length, _O(N²)_, this method is sub-quadratic/linear in N (_O(N)_). We’ll see later why & how.
- **Exact** — meaning it’s not an approximation of the attention mechanism (like e.g. sparse, or low-rank matrix approximation methods) — its outputs are the same as in the “vanilla” attention mechanism.
- **IO aware**

### Explanation [Background]

Let’s expand on this IO awareness part a bit more. “IO” is the reason more FLOPS doesn’t necessarily translate into longer wall-clock time.

Over the years GPUs have been adding compute capacity (FLOPS) at a faster pace than increasing the memory throughput (TB/s).

**It doesn’t matter if you can compute at exaFLOPS speeds if there is no data to be processed.** These 2 need to be closely aligned, and since the hardware lost that balance we have to make our software compensate for it.

Depending on this ratio between computation and memory accesses, operations can be classified as either:

- **compute-bound** (example: matrix multiplication)
- OR **memory-bound** (examples: elementwise ops (activation, dropout, masking), reduction ops (softmax, layer norm, sum, etc.)…)

*Note on the terminology: this ratio is commonly measured by the **arithmetic intensity**, which is the number of arithmetic operations per byte of memory access.*

It turns out **attention is** (on current AI accelerators) **memory-bound**.

Why?

Because it “mostly consists of elementwise ops” or more accurately the arithmetic density of attention is not very high.
![[Pasted image 20240614091837.png]]A100 GPU has **40–80GB** of high bandwidth memory (HBM, the thing that gives you lovely CUDA OOMs) with a bandwidth of **1.5–2.0 TB/s** and **192KB** of on-chip SRAM per each of 108 streaming multiprocessors with bandwidth estimated around **19TB/s.**

Standard attention schema is the following: 
![[Pasted image 20240614092110.png]]The lowest hanging fruit is to **remove redundant HBM reads/writes**.
Why write **_S_** back to HBM only to (re)load it again in order to compute the softmax? Let’s keep it in SRAM instead, perform all of the intermediate steps, and only then write the final result back to HBM

This is what compilers folks refer to as **“kernel fusion”**, one of the most important low-level optimizations in deep learning:
![[Pasted image 20240614094901.png]]==A== ==**_kernel_**== ==is basically a fancy way of saying “a GPU operation”.
**_Fusion_** means you’re fusing/combining multiple ops together

So, you are loading from the HBM only **once,** you execute the fused op, and only then write the results back. By doing this you reduce the communication overhead.

One final piece of terminology you’ll find floating around is **“materialization”**. 
It refers to the fact that in the above standard attention implementation, we’ve **allocated** full **NxN** matrices (**_S_**, **_P_**). We’ll soon see that that’s the bottleneck flash attention directly tackles reducing the memory complexity from _O(N²)_ to _O(N)._


Flash attention basically boils down to 2 main ideas:

1. **Tiling** (used during both forward & backward passes) — basically chunking the NxN softmax/scores matrix into blocks.

2. **Recomputation** (used in the backward pass only — if you’re familiar with activation/gradient checkpointing, this will be trivial to understand)

![[Pasted image 20240614095324.png]]

### FlashAttention [Main algorithm]

The main hurdle in getting the tiling approach to work is softmax. 
**In particular, the fact that softmax couples all of the score columns together. 
Here is how we compute the _i-th_ output of a softmax.**:
![[Pasted image 20240614095608.png]]
The denominator is the issue.

**To compute how much a particular _i-th_ token from the input sequence pays attention to other tokens in the sequence you’d need to have all of those scores readily available (denoted here by _z_j_) in SRAM. But let me remind you: SRAM is severely limited in its capacity. You can’t just load the whole thing. N (sequence length) can be 1000 or even 100.000 tokens. So _N²_ explodes fairly quickly.**

```
So here’s the trick, we can actually chop the softmax computation down into smaller blocks and still end up with precisely the same result.
```
![[Pasted image 20240614095823.png]]`These numbers are, at least for now, incorrect.But bear with me, through iterations, we’ll “converge” to a correct result`

> Note: **you can ignore the m(x)** part, at least for now while we’re still in Plato’s world of ideas. Its purpose is solely to avoid numerical instabilities. On some hypothetical hardware from the future that’s more precise (e.g. we represent our data using more bits) this would not be needed. **m(x)** does not change the final result in any way.

We can combine those per-block partial softmax numbers in a smart way such that the final result is actually correct. Here is the main idea:
![[Pasted image 20240614100444.png]]So basically, in order to compute the softmax for the scores belonging to the first 2 blocks (of size _B_), you have to keep track of 2 statistics for each of the blocks: **_m(x)_** (maximum score) and **_l(x)_** (sum of exp scores).

And then you can seamlessly fuse them together using the normalizing coefficients.

This logic continues recursively all the way up to the last, _(N/B)-th,_ block, at which point you have the N-dimensional correct softmax output!

> Note: the algo below assumes we have a batch of size 1 (i.e. single sequence) and a single attention head, we’ll easily scale it up later (by simply parallelizing across GPU’s streaming multiprocessors — more on that later). Also we ignore dropout & masking for the time being, trivial to add it later.


![[Pasted image 20240614101221.png]]
#### Steps 
S**tep 0:** HBM’s capacity is measured in GBs (e.g. RTX 3090 has 24 GBs of VRAM/HBM, A100 has 40–80 GB, etc.) so allocating **_Q_**, **_K_**, and **_V_** is not an issue.

**Step 1:** Let’s compute the row/column block sizes. Why _ceil(M/4d)_? Because query, key, and value vectors are d-dimensional, and, we also need to combine them into the output d-dimensional vector. So this size basically allows us to max out SRAM capacity with _q_, _k_, _v_, and _o_ vectors.

Toy example: assume M = 1000, d = 5. In this example, the block size is (1000/4*5) = 50. So in this example, we would load blocks of 50 _q, k, v, o_ vectors at a time, to make sure we’re reducing the number of reads/writes between HBM/SRAM.

**Step 2:**

We initialize the output matrix **_O_** with all 0s. It’ll act as an accumulator hence that init value. Similarly for **_l_** (remember: its purpose is to hold the cumulative denominator for the softmax - the sum of exp scores). **_m_** (that holds row-wise maximum scores) is initialized with _-inf_ because we’ll be doing a max operator over it so whatever the first block’s max is — it’ll certainly be larger than _-inf —_ hence this is the natural init value.

**Step 3:
We split the **_Q, K,_** and **_V_** into blocks using the block sizes from Step 1. 

**Step 4**:
Similarly split **_O, l, m_** into blocks (same block size as **_Q_**)

Step 5:
Let’s start looping across the columns i.e. across key/value vectors (**outer loop** in the diagram above).

Step 6:
Let’s load the **_K_j_** and **_V_j_** blocks from HBM to SRAM. Remember because of the way we constructed the block sizes we still have 50% of the SRAM unoccupied at this point in time (dedicated to **Q** and **O**)

Step 7: 
Start the **inner loop** across the rows i.e. across query vectors (again, see the diagram).

Step 8:
Load **_Q_i_** (_B_r x d_) and **_O_i (_**_B_r x d_**_)_** blocks, as well as **_l_i_** (_B_r_) & **_m_i (_**_B_r_**_)_** into SRAM.

How do **_l_i_** & **_m_i_** fit into the SRAM (including all of the intermediate variables) when we computed block size in such a way that we only have enough space for **_K_j_**, **_V_j_**, **_Q_i_** & **_O_i_**? I think the answer is: registers (see [this CUDA video series](https://www.youtube.com/watch?v=4APkMJdiudU&list=PLC6u37oFvF40BAm7gwVP7uDdzmW83yHPe) to get some intuition on GPU memory hierarchy). But I might be wrong, 

Step 9:
Compute the dot product between **_Q_i_** (_B_r x d_) and **_K_j_** transposed (_d x B_c_) to get the scores (_B_r x B_c_). As you can see we don’t have the whole _NxN_ **_S_** (scores) matrix “materialized”. Only a fraction of it (**_S_i_j_**)!

Step 10: 
Compute **_m~_i_j_**, **_l~_i_j_**, and **_P~_i_j_** using the scores computed in the previous step. It’s trivial.

**_m~_i_j_** is computed row-wise, find the max element for each of the above rows.

We get **_P~_i_j_** by applying elementwise ops:

1. Normalization — take the row max and subtract it from row scores
2. Exp

**_l~_i_j_** is simply a row-wise sum of the matrix P.

Step 11:
Compute **_m_new_i_** and **_l_new_i_**.
**_m_i_** contains row-wise maximums for all of the blocks that came before. **_m~_i_j_** contains the row-wise maximums for the current block . To get the **_m_new_i_** we just have to apply a max between **_m~_i_j_** & **_m_i_**. Similarly for **_l_new_i_** (it additionally requires multiplying by coefficients as we saw previously in _formula 2_).


Step 12 (Most important step):
![[Pasted image 20240614105637.png]]
This is the hardest part of the algorithm but still not that complicated, esp. once you internalize the _formulas 1 & 2_ for partial softmax computation.

Let’s break down the **_diag(l)_** part first.

It basically just allows us to do row-wise scalar multiplication in a matrix form. If you have a list of scalars **_s_** (_N_) and a matrix **_A (_**_NxN_**_)_**, if you do **_diag(s)_*****_A_** you’re basically doing elementwise multiplication of rows of **_A_** with those scalars


So what the 1st term of step 12 does (underlined in green) is it updates the current softmax estimate for the blocks before the current block in the same row of blocks. In case j=1 (that is the first block in this row) the 1st term will be 0 and we’ll just end up with the 2nd term.
The multiplication of the 1st term by **_diag(l_i)_** is there to cancel the division by that same constant from the previous iteration (this constant is hidden inside of **_O_i_**).

The 2nd term of the expression (underlined in yellow) doesn’t require this canceling of terms because as you can see we’re directly multiplying the **_P~_i_j_** matrix with the block of **_V_** vectors (**_V_j_**).

The **_e^x_** terms are there to modify the matrix **_P~_i_j_** & **_O_i_** by canceling out the **_m_** from the previous iteration and instead updating it with the latest estimate (**_m_new_i_**) that contains the row-wise max so far.

Step 13: 
Write the newest cumulative statistics (**_l_i_** & **_m_i_**) back to HBM. Notice these are of dimension _B_r_.

Step 14.15,16:
Once the nested for loop is over, **_O_** (Nxd) will contain the final result: attention-weighted value vectors for each of the input tokens

This algorithm can easily be extended to “block-sparse FlashAttention”, a sparse attention algorithm that is 2–4 faster than even FlashAttention, scaling up to a sequence length of 64k! The idea is we use a block form mask matrix and we simply skip certain loads/stores from the above nested for loop and by doing so we can save proportionally to the sparsity coefficient.


### Scaling

Let’s start with the low-hanging fruit. Extending the implementation we saw to support _batch_size_ > 1 and the _num_heads_ > 1 is actually not that hard.

So far the algorithm we saw is basically handled by a single **_thread block_** (CUDA programming lingo). This thread block is executed on a single **_streaming multiprocessor_** (SM) (e.g. there are 108 of these on A100). To parallelize our computation we just run _batch_size_ * _num_heads_ threadblocks in parallel on different SMs. The closer that number is to the number of available SMs on the system the higher the utilization will be (ideally a multiple as each SM can run multiple thread blocks).

What happens when that number is bigger than the number of available SMs? I’m not sure but I assume there is a queue that keeps track of the waiting kernels (update: [apparently](https://www.youtube.com/watch?v=xwbD6fL5qC8&t=760s&ab_channel=TomNurkkala) the CUDA runtime takes care of that and it is using some sort of queues to implement that logic).

The backward pass relies on the same set of concepts + **_recomputation_**.

To demonstrate the concept of recomputation I’ll use the example of “_activation/gradient checkpointing_” method.

We know that we need to have the activations computed during the forward pass readily available during the backward pass in order to compute the gradients w.r.t. our loss function.

The trick here is to not store them during the fwd pass (as they have a huge memory footprint), but instead, recompute them de novo during the backward pass. There is a built-in **tradeoff** here: we’re slowing down the backward pass in order to reduce the memory footprint.

> Note: This tradeoff is a spectrum, e.g. you can store the activations every _n_ layers, and then when computing the activations for the i-th layer you don’t have to start from the input but instead from the closest stored activations.

The same concept of recomputation is re-used here — but with a twist! Luckily for the flash attention, we don’t have to sacrifice neither runtime nor memory!

By storing the output **_O_** (_Nxd_) and the softmax normalization statistics (_N_) we can recompute the attention matrices **_S_** (_NxN_) and **_P_** (_NxN_) in the backward pass directly from blocks of **_Q_**, **_K_**, and **_V_** (_Nxd_) in SRAM! Thus keeping the memory at _O(N)_.




## Model File Sharding

The original model file is usually sharded into multiple chunks, typically 10GB each.
Our execution processes layer by layer. Each layer is only 1.6GB. If we load based on the original 10GB shards, every layer execution will require reloading the entire 10GB file but only using 1.6GB

This process wastes a lot of memory for loading and disk reading. Disk reading speed is actually the slowest bottleneck in the whole inference process, so we want to minimize it as much as possible.
Therefore, we first **pre-process the original HuggingFace model file and shard it by layers**.

**Safetensor ensures the storage format and in-memory format match closely, and uses memory mapping for loading to maximize speed.**

## Meta device

Meta device is a **virtual device** designed specifically for running ultra large models. **When you load a model via meta device, the model data is not actually read in, only the code is loaded. Memory usage is 0.**

You can dynamically transfer parts of the model from meta device to a real device like CPU or GPU during execution. Only then is it actually loaded into memory.

Using init_empty_weights() allows model loading via meta device.

```
from accelerate import init_empty_weights
with init_empty_weights():
    my_model = ModelClass(...)
```

## AirLLM inference

```
from airllm import AirLLMLlama2

MAX_LENGTH = 128
# could use hugging face model repo id:
model = AirLLMLlama2("garage-bAInd/Platypus2-70B-instruct")

# or use model's local path...
#model = AirLLMLlama2("/home/ubuntu/.cache/huggingface/hub/models--garage-bAInd--Platypus2-70B-instruct/snapshots/b585e74bcaae02e52665d9ac6d23f4d0dbc81a0f")

input_text = [
        'What is the capital of United States?',
    ]

input_tokens = model.tokenizer(input_text,
    return_tensors="pt", 
    return_attention_mask=False, 
    truncation=True, 
    max_length=MAX_LENGTH, 
    padding=True)
           
generation_output = model.generate(
    input_tokens['input_ids'].cuda(), 
    max_new_tokens=20,
    use_cache=True,
    return_dict_in_generate=True)

output = model.tokenizer.decode(generation_output.sequences[0])

print(output)

```