## 🍭 Блендинг

Ты уже проанализировал данные, нагенерировал **gold** фичей,  отфильтровал мусорные признаки и затюнил все версии бустингов и сеток. Однако тебе все еще не хватает. Что делать? К счастью, это еще не конец, скор можно еще значительно улучшить, если смешать прогнозы всех имеющихся моделей.

- Основная идея блендинга в том, чтобы взять от каждого алгоритма лучшее, совместив несколько разных ML-моделей. 
- За счет такого объединения улучшается качество и увеличивается обобщающая способность финальной модели.
- Помимо этого ваша модель становится более стабильной, что позволяет не слететь на приватном лидерборде.
- Особенно хорошо накидывает блендинг, если смешиваемые **модели имеют разную природу.** _Например_, нейронные сети, KNN и решающие деревья. В этом случае они выучивают разные зависимости и хорошо дополняют друг друга.

🍏 Самый просто способ сделать блендинг — это усреднить предсказания или взвесить ответы. Буквально строчка кода:
```
ensemble = catboost_model * w1 + xgboost_model * w2 + lgbm_model * w3 
```
### 🤹‍♀️ Основные подходы к блендингу
#### **🧩 Голосование и усреднение**

Для задачи регрессии самый простой вариант — объединить несколько моделей и взять среднее их предсказаний.

Для задачи классификации есть два базовых подхода:

1) `Hard Voting` — в качестве итогового предсказания выбирается самый частый класс среди предсказанных базовыми моделями.   
2) `Soft Voting` — здесь мы **основываемся уже на вероятностях**, которые были предсказаны каждой моделью. Для получения финального решения нужно сложить вероятности для каждого класса и потом взять класс с максимальной суммой.

#### **🧩 Голосование и усреднение с весами**

Проблема обычного усреднения или голосования в том, что мы никак не учитываем точность, с которой каждая из базовых моделей работает. В таком случае модель, которая работает с точностью 90% и с точностью 60%, будет учтена с одинаковым весом в ансамбле. 🤔    
  
💡 Отсюда вытекает модификация предыдущей техники — давайте в качестве финального предсказания брать не просто среднее, а взвешенное среднее:

```python
final_preds = w1 * model1_preds + w2 * model2_preds + w3 * model3_preds
# где w1 + w2 + w3 = 1
```

 🍏 Аналогичный подход можно применить и для `Hard Voting`, где какой-то модели будет даваться несколько голосов, вместо одного.

**To calculate weights: **
```python
total_score = sum(scores)
weights = [round(score / total_score,7) for score in scores]
```

#### Rank Averaging🥇🥈🥉
Рассмотрим задачу ранжирования: Пусть у нас есть покупатели, и нужно отранжировать товары по вероятности того, что данный товар интересен покупателю.

**Разная калибровка моделей**

Допустим, у нас уже есть 4 разных товара, один покупатель и несколько моделей (для примера возьмем две):

1) первая модель предсказывает в качестве релевантности товаров числа  0.3, 0.4, 0.5 и 0.2;  
2) вторая модель — 0.7, 0.75, 0.79 и 0.68, соответственно.

Несмотря на то что числа она выдает разные, если мы их отсортируем по убыванию, то получим одинаковые последовательности товаров: №3, №2, №1, №4. А раз у нас задача ранжирования, то и значение метрики у обеих моделей будет одинаково.

**Нормализация предсказаний**

Как вы видите, несмотря на то что модели делают предсказания в разных числовых диапазонах, они по факту оказываются одинаково хорошо. Но можем ли мы просто усреднить их предсказания во время блендинга? Нет. Если у нас модели по-разному откалиброваны, то усреднение может не только не сыграть вам на руку, но и ухудшить скор, так как одна модель будет "перетягивать" на себя общее предсказания. Например, в нашем случае модель №1 будет иметь решающую роль, так как разница между ее предсказаниями сильно больше, чем у второй модели.

Для того чтобы избежать этой проблемы и правильно учитывать предсказания обеих моделей, мы можем рассчитать ранг каждого предсказания — каким по счету будет наш товар (у данного покупателя), если мы отсортируем товары по возрастанию их релевантности. 

|Номер товара|Предсказания 1|Предсказания 2|Ранг 1|Ранг 2|
|---|---|---|---|---|
|1|0.3|0.95|2|2|
|2|0.4|0.9|3|1|
|3|0.5|0.99|4|4|
|4|0.2|0.98|1|3|

Теперь мы можем рассчитать отнормированные предсказания `norm_pred = rank / n`.

|Номер товара|Предсказания 1|Предсказания 2|Ранг 1|Ранг 2|Нормированные 1|Нормированные 2|
|---|---|---|---|---|---|---|
|1|0.3|0.95|2|2|0.5|0.5|
|2|0.4|0.9|3|1|0.75|0.25|
|3|0.5|0.99|4|4|1|1|
|4|0.2|0.98|1|3|0.25|0.75|

Теперь, чтобы сблендить модели, нам нужно просто взять среднее значение отнормированных предсказаний, и таким образом мы сможем избавиться от проблемы разной калибровки моделей при блендинге.

**Примечание**: Чтобы сделать предсказания на основе новых данных, необходимо понять, какой ранг был бы у данного предсказания, если бы оно было частью тестовой выборки. Для этого можно просто найти элемент в тестовой выборке, который ближе всего находится к нашему, и взять его ранг.
### 🐲 Принципы блендинга (просто и логично)

- 🦑 Не бленди, пока не выжал максимум из моделей по отдельности
- 🐳 Чем различнее и сильнее модели, тем эффективнее блендинг (⚠️ эффективность)
- 🐙 При равной точности, ансабль побеждает соло-модель на привате (⚠️ стабильность)
- 🦐 Чем раньше проверишь эффект от блендинга, тем эффективне будет стратегия
- 🐠 Блендить можно с разными весами, пропорционально скорам или разным фичам
- 🐋 Блендинг по фолдам и чекпоинтам обучения - это тоже блендинг
- 🐡 Блендинг по сидам - это стабилизирующий блендинг
- 🐬 Против блендинга только больший блендинг

## 💎 Стекинг
В прошлый раз мы познакомились с технологией блендинга, где нужно было комбинировать предсказания моделей простыми методами. Пришло время познакомиться с более сильной техникой смешивания моделей - стекинг. Давайте вместо обычного усреднения предсказаний модели обучим отдельную метамодель, которая будет на вход принимать предсказания базовых моделей и по ним предсказывать целевую переменную.

### **Выбираем метамодель**
Самым базовым вариантов в качестве метамодели выступает `Linear Regression`. Она не склонна к переобучению и  просто правильно взвешивает предсказания модели. 

Однако `Linear Regression` не может учитывать нелинейные зависимости, так что частенько для метамодели выбираются такие алгоритмы, как `CatBoost, XGBoost, Random Forest` и другие.

В принципе, не обязательно даже ограничиваться одной метамоделью, можно обучить несколько разных метамоделей и потом просто усреднить их значения. Хотя на практике данный подход встречается реже, чем предыдущие два в силу своей накрученности.

### Стекинг с фичами
**1. Добавляем ранги**

Помните `Ranking Average` - когда мы считаем относительное расположение предиктов по каждой модели и потом их усредняем. Так вот, на самом деле, ранги можно добавить еще как признаки в метамодель. Как показывает практика, это лучше, чем использовать отдельно ранги или отдельно предсказанные вероятности.

**2. Считаем квадратичные зависимости**

На самом деле, нам никто не мешает скомбинировать предсказания базовых моделей. Например, путем их попарного перемножения и вычитания. Далее эти комбинации также добавляются к признаковому пространству для метамодели. 

Такой подход улучшил качество во многих экспериментах, наиболее заметно в соревновании по моделированию женского здоровья от `DrivenData`.

**3. Добавляем признаки от базовых моделей**

В качестве дополнительных признаков можно добавить все те признаки, на которых учились базовые модели. Это может помочь метамодели лучше комбинировать предсказания базовых моделей.

В качестве модификации этого подхода иногда добавляют взвешенные признаки, где в качестве весов выступают вероятности базовых моделей. То есть если у нас изначально было n признаков и m базовых моделей, то на вход финальной модели будут подаваться `n * m` признаков вида `feature[i] * preds[j]`.
### Простой пример стекинга
``` python
class Blender:
    def __init__(self, base_models, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model

    def fit(self, X_train, y_train, X_hold, y_hold):
        preds = []
        for model in self.base_models:
            model.fit(X_train, y_train)
            pred = model.predict(X_hold)
            preds.append(pred)
        result = np.column_stack(preds)
        self.meta_model.fit(result, y_hold)
     
    def predict(self, X_test):
        test_preds = np.column_stack([model.predict(X_test) for model in self.base_models])
                
        preds = self.meta_model.predict(test_preds)
        return preds
```

### Стекинг с фолдами
Пока  мы обучали метамодели на отдельно отложенной (`holdout`) выборке, но в таком случае мы не использовали все доступные нам данные по максимуму: при стандартном разбиении 80/20, базовые модели обучаются только на 80% данных, а метамодель - вообще только на 20%. Однако с помощью кросс-валидации это можно исправить.

**Обучаем базовые модели**

Пусть мы используем 4-x фолдовую кросс-валидацию при обучении базовых моделей. Тогда каждая модель обучается на 3-х фолдах и на последнем валидируется. Идея заключается в том, чтобы для каждого примера из обучающий выборки сформировать предсказания с помощью базовой модели, которая при обучении его не видела. То есть если мы сейчас обучаемся на `fold_1`, `fold_3` и `fold_4`, то мы валидируемся по **`fold_2`** и на него же делаем предсказания. Таким образом, каждая из моделей сделает предсказания для фолда, на котором не обучалось, и объединение таких предсказаний мы сможем использовать для обучения метамодели.

Для того чтобы сделать предсказания на `X_test`, можно либо усреднить предсказания моделей, полученных во время кросс-валидации, либо обучить модель еще раз, но уже на всем `X_train` (без CV).

```
def GetPreds(model, X, y, n_fold=5):
    folds = KFold(n_splits=n_fold)
    preds = np.empty(len(X), float)

    for train_indices, val_indices in folds.split(X, y):
        X_train,y_train = X.iloc[train_indices],y[train_indices]
        X_val,y_val = X.iloc[val_indices],y[val_indices]
        model.fit(X_train,y_train)
        preds[val_indices] = model.predict(X_val)
    return preds.reshape(-1, 1)
```



##  🛠 Автоматический стекинг

Как вы уже могли заметить, чем больше моделей вы стекаете, тем больше у вас разрастается код, а кол-во беспорядка в нем растет по экспоненте. Плюс к этому становится тяжело менеджерить предобработку признаков для отдельных моделей, т. к. какие-то модели не работают с категориальными признаками, где-то нужно заполнять `NaN`, а где-то лучше их оставить и т. д.

Но есть специальные инструменты, которые позволяют сделать это элегантно и даже более эффективно. Да еще и меньшим числом строк кода!  `sklearn.Pipelines` - способ упаковать ваш процесс обучения и инференса от `Feature Engineeringа` до стекинга 10 моделей в один пайплайн.

### First, preprocess the data:
```
from sklearn import preprocessing
data = pd.read_csv('../data/quickstart_train.csv')

categorical_features = ['model', 'car_type', 'fuel_type']

for cat in categorical_features:
    lbl = preprocessing.LabelEncoder()
    data[cat] = lbl.fit_transform(data[cat].astype(str))
    data[cat] = data[cat].astype('category')
    
# значения таргета закодируем целыми числами
class_names = np.unique(data['target_class'])
data['target_class'] = data['target_class'].replace(class_names, np.arange(data['target_class'].nunique()))
```
Split train/test:
```
cols2drop = ['car_id', 'target_reg', 'target_class']
categorical_features = ['model', 'car_type', 'fuel_type']
numerical_features = [c for c in data.columns if c not in categorical_features and c not in cols2drop]

X_train, X_val, y_train, y_val = train_test_split(data.drop(cols2drop, axis=1), 
                                                    data['target_class'],
                                                    test_size=.25,
                                                    stratify=data['target_class'],
                                                    random_state=42)
```
### Second, declare the models:
#### CatBoost
```
params_cat = {
             'n_estimators' : 700,
              'depth' : 3,
              'verbose': False,
              'use_best_model': True,
              'cat_features' : categorical_features,
              'text_features': [],
              'border_count' : 64,
              'l2_leaf_reg' : 1,
              'bagging_temperature' : 2,
              'rsm' : 0.51,
              'loss_function': 'MultiClass',
              'auto_class_weights' : 'Balanced', #try not balanced
              'random_state': 42,
              'use_best_model': False,
              # 'custom_metric' : ['AUC', 'MAP'] # Не работает внутри sklearn.Pipelines
         }
cat_model = cb.CatBoostClassifier(**params_cat)
```
#### LightGBM
```
categorical_features_index = [i for i in range(data.shape[1]) if data.columns[i] in categorical_features]
params_lgbm = {
    "num_leaves": 200,
    "n_estimators": 1500,
    "min_child_samples": None,
    "learning_rate": 0.001,
    "min_data_in_leaf": 5,
    "feature_fraction": 0.98,
    'reg_alpha' : 3.0,
    'reg_lambda' : 5.0,
    'categorical_feature': categorical_features_index
}
lgbm_model = lgbm.LGBMClassifier(**params_lgbm)
```
#### XGBoost
```
params_xgb = {
    "eta": 0.05,
    'n_estimators' : 1500,
    "max_depth": 6,
    "subsample": 0.7,
    'min_child_weight' : 0.1,
    'gamma': .01,
    'reg_lambda' : 0.1,
    'reg_alpha' : 0.5,
    "objective": "reg:linear",
    "eval_metric": "mae",
    'tree_method' : 'hist', # Supported tree methods for cat fs are `gpu_hist`, `approx`, and `hist`.
    'enable_categorical' : True
}
xgb_model = xgb.XGBClassifier(**params_xgb)
```
### Third, pipeline:
```
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_selector

# Вспомогательные элементы для наполнения пайплайна
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, OneHotEncoder, MinMaxScaler

# Некоторые модели для построения ансамбля
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC

# Добавим визуализации
import sklearn
sklearn.set_config(display='diagram')
```

First, make a transformer for each data-type:
```
# заменяет пропуски самым частым значением и делает ohe
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy='most_frequent')),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))])

# заменяет пропуски средним значением и делает нормализацию
numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer()),
    ("scaler", StandardScaler())
])

preprocessor = ColumnTransformer(transformers=[
    ("numerical", numerical_transformer, numerical_features),
    ("categorical", categorical_transformer, categorical_features)])

preprocessor
```
Let's start the ensamble:
```
# список базовых моделей
estimators = [
    ("ExtraTrees",  make_pipeline(preprocessor, ExtraTreesClassifier(n_estimators = 10_000, max_depth = 6, min_samples_leaf = 2, 
                                                              bootstrap = True, class_weight = 'balanced', # ccp_alpha = 0.001, 
                                                              random_state = 75, verbose=False, n_jobs=-1,))),
    ("XGBoost", xgb_model),
    ("LightGBM", lgbm_model),
    ("CatBoost", cat_model),
    ("Random_forest",  make_pipeline(preprocessor, RandomForestClassifier(n_estimators = 15_000, max_depth = 7, 
                                                              min_samples_leaf = 2,
                                                              warm_start = True, n_jobs=-1,
                                                              random_state = 75, verbose=False))),
]

# в качестве мета-модели будем использовать LogisticRegression
meta_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(verbose=False),
)

stacking_classifier = meta_model
stacking_classifier.fit(X_train, y_train)
```
Now let's check the accuracies of the models and also how they correlate with each other:
```
corr_df = pd.DataFrame()

for model, (name, _) in zip(stacking_classifier.estimators_, stacking_classifier.estimators):
    print(name, 'accuracy: ', round(accuracy_score(model.predict(X_val), y_val), 4))
    corr_df[name] = model.predict(X_val)
corr_df.corr().style.background_gradient(cmap="RdYlGn")
```
If something correaltes too much with other models -> it hinders the performance , try to take it away.
### Commentary on sklearn implementation

- 📈 Да, скор ансамбля вырос, но есть много **"но"** у этой реализации
- ⚠️ Тут в качестве мета-модели использовалась `LogisticRegression`, что по сути является обычным блендингом, но с кросс-валидацией.
- 🧩 Слабые или похожие модели мешают ансамблю поднять скор (Если убрать `RandomForest`, то скор поднимется)
- 🍏 Стекинг можно усложнить, подавая мета-модели еще признаки, при этом используя более сложную meta-модель.
- 🤔 Тогда в зависимости от свойств объекта, мета-модели, такие как `RandomForestClassifier`, могут принимать решение оптимальнее.
- ☹️ В рамках `pipeline` в `sklearn` это сделать сложнее. Надо взять что-то другое.
- Не все можно запихнуть в `pipeline`. Например, `eval_set` для `early-stopping` или класс `train` от `LightGBM`
### Writing your own transformer: 
```
import sklearn import pandas as pd 
from sklearn.base import BaseEstimator, TransformerMixin 
class TextTransformer(BaseEstimator, TransformerMixin): 
	def __init__(self, stop_words): 
		self.stop_words = stop_words 
	def fit(self, X, y = None): 
		return self 
	def filter_text(self, text): 
		word_tokens = text.split() 
		word_tokens = [w for w in word_tokens if not w in self.stop_words] 
		word_tokens = [get_normal_string(w) for w in word_tokens] 
		return ' '.join(word_tokens) 
	def transform(self, X): 
		for col in range(X.shape[1]): 
			X[:, col] = [self.filter_text(el) for el in X[:, col]] 
		return X
```


### PyStackNet

```

from pystacknet.pystacknet import StackNetClassifier  

model=StackNetClassifier(models, metric="auc", folds=4, restacking=False,use_retraining=True, use_proba=True, 
						random_state=12345,n_jobs=1, verbose=1) 

# Наконец, чтобы обучить, нужно воспользоваться стандартными функциями fit/predict 
model.fit(x,y) preds = model.predict_proba(x_test)
```
Model list:
```
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier 
from sklearn.linear_model import LogisticRegression
models = [ ######## First level ######## 
			[ RandomForestClassifier( n_estimators=100, max_features=0.5, random_state=1, ),
			 ExtraTreesClassifier( max_depth=5, max_features=0.5, random_state=1, ), 
			 GradientBoostingClassifier( learning_rate=0.1, random_state=1, ),
			  LogisticRegression(random_state=1), ],
			   ######## Second level ######## 
			[ RandomForestClassifier( n_estimators=200, criterion="entropy", random_state=1, ) ], 
		]
```