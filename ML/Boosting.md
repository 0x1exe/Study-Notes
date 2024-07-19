#### Общее
 - ФИКСИРОВАТЬ RANDOM_STATE
 - Иметь early_stop
 - При KFold обучении суммировать и усреднять предсказания по фолдам
##### KFold training code
```
def fold_train(X,
               y,
               model,
               params,
               n_folds,
               cat_features,
               ):

  kf = KFold(n_splits= n_folds, shuffle=True, random_state=RANDOM_STATE)
  scores=[]
  models=[]
  print(f"========= INITIATING {model.__name__} =========")
  for num_fold, (train_idx,test_idx) in enumerate(kf.split(X)):
    X_train,y_train = X.iloc[train_idx],y.iloc[train_idx]
    X_test,y_test =  X.iloc[test_idx],y.iloc[test_idx]
    if params is not None:
      clf = model(**params)
    else:
      clf = model()
      
    if model.__name__ == "CatBoostRegressor":
      train_set = Pool(data=X_train,label=y_train,cat_features=cat_features)
      test_set = Pool(data=X_test,label=y_test,cat_features=cat_features)
      clf.fit(train_set,
              eval_set=test_set,
              verbose=0)
              
    if model.__name__ == "LGBMRegressor":
   train_set=Dataset(X_train,y_train,categorical_feature=cat_features,free_raw_data=False)

test_set=Dataset(X_test,y_test,categorical_feature=cat_features,free_raw_data=False,reference= train_set)
      #evals = {}
      clf = lgb.train(
          params,
          train_set=train_set,
          valid_sets=[test_set],)
          #callbacks=[lgb.record_evaluation(evals)])
          
    if model.__name__ == "XGBRegressor":
      train_set = DMatrix(X_train,label=y_train,nthread=-1,enable_categorical=True)
      test_set = DMatrix(X_test,label=y_test,nthread=-1,enable_categorical=True)
      clf = xgb.train(params,
                    dtrain=train_set,
                    evals=[(train_set, 'dtrain'), (test_set, 'dtest')],
                    verbose_eval=0)
      X_test=test_set

  
    y_pred=clf.predict(X_test)
    score=mean_squared_error(y_pred,y_test)
    models.append(clf)
    scores.append(score)

    print(f'=== Fold: {num_fold} === score: {score} ===')
    
  final_score = np.mean(scores,dtype='float16') - np.std(scores,dtype='float16')
  print(f'Final score: {final_score}')

  best_model = models[np.argmin(scores)]
  return final_score,best_model
```
##### GridSearchCode
```
def tuning_hyperparams(algorithm,
                       X,
                       y,
                       init_params,
                       grid_params,
                       n_iter,
                       cv=3,
                       fit_params = None,
                       random_state=2023,
    ):
    estimator = algorithm(**init_params)
    model = RandomizedSearchCV(estimator=estimator,
                               param_distributions=grid_params,
                               n_iter=n_iter,
                               cv=cv,
                               scoring='neg_root_mean_squared_error',
                               n_jobs=-1,
                               verbose=0,
                               random_state=RANDOM_STATE

    )
    model.fit(X, y)
    return model.best_params_ | init_params
```
#### CatBoost
##### Parameter notes [Pt1]
- **Принимает категориальные фичи** сразу без всякой предварительной обработки.
- Чтобы перенести обучение с **`CPU`** на **`GPU`** достаточно поменять значение 1 параметра, без установки доп.пакетов или специальных версий, как в других библиотеках
- Даже с дефолтными параметрами выдает хорошую точность модели
    - Основные параметры не константные, а **подбираются самой библиотекой**, в зависимости от размера входных данных.
- Может принимать текстовые признаки, эмбеддинги, временные признаки.
- Без дополнительных манипуляций и оберток встраивается в стандартные пайплайны (например, `sklearn`).
- Идет в комплекте с "батарейками": `feature_selection`, `object_selection`, `cross_validation`, `grid_search` и пр
- l2_leaf_reg : штрафует модель предотвращая подгонку под шум в данных
- min_data_in_leaf
- max_leaves:  меньше листов - сложнее модель
- subsample: подвыборка для каждого дерева приводит к снижению подгонки под шум
- colsample_bylevel: использование лишь части признаков при разбиении
- Предварительная квантилизация : разбиение непрерывного признака на дискретные бины, полезна при работе с данными с большим разбросом. Вероятна потеря информации, следует увеличивать кол-во разбиений для голд признаков
##### Parameter notes [Pt2]

**Базовые параметры**
- `iterations` (синонимы `num_boost_round`, `n_estimators`, `num_trees`) - максимальное количество деревьев, используемых в модели (по умолчанию **`1000`**).  
    Значение может быть ниже заданного, если используются другие параметры, накладывающие ограничение на количество деревьев;
- `learning_rate` или `eta` – скорость обучения, которая определяет, насколько быстро или медленно модель будет учиться. Значение по умолчанию обычно равно **`0.03`**.
- `depth (max_depth)` - глубина дерева (по умолчанию 6, максимальное значение - 16);
- `cat_features` - список наименований категориальных признаков;

**Режим обучения**
- `loss_function` или `objective`- функция потерь, которую надо минимизировать при обучении. Есть показатели для регрессии (среднеквадратичная ошибка), есть для классификации (`logloss`).
- `eval_metric` - валидационная метрика, используемая для обнаружения переобучения и ранней остановки;
- `custom_metric` - отслеживаемые метрики. Лучшие метрики в процессе обучения можно вернуть методом `get_best_score`;
- `early_stopping_rounds` - определяет число итераций до остановки, если на их протяжении метрика качества не улучшалась по сравнению с оптимальной;
- `use_best_model` - если **`True`**, в результате обучения с ранней остановкой будет возвращаться модель, полученная на итерации (количеством деревьев) с лучшей метрикой на валидационной выборке;

**Регуляризация модели, ускоряющие и обобщающие модель**
- `l2_leaf_reg` (или `reg_lambda`) – коэффициент при члене регуляризации **`L2`** функции потерь. Значение по умолчанию – **`3.0`**.
- `min_data_in_leaf (min_child_samples)` - минимальное количество обучающих сэмплов в листе.
- `max_leaves (num_leaves)` - максимальное количество листьев в дереве.
- `subsample` - часть исходной выборки, отбираемая для обучения каждого дерева;
- `colsample_bylevel` - доля признаков, используемая для отбора на каждом сплите;
- `max_bin` - максимальное число бинов, на которые разбиваются признаки

**Полезно использовать**
- `random_seed` или `random_state` – инициализирующее значение для используемого генератора случайных чисел, чтобы обеспечить воспроизводимость эксперимента;
- `task_type` - используемое для вычислений устройство (**`CPU`**, **`GPU`**);
- `thread_count` - число используемых для обучения потоков (по умолчанию = **`-1`**, что означает - все ядра процессора);
- `verbose` - объем выводимой информации (**`False`** - подавляем сообщения).

**Балансировка классов**
- `class_weights` - веса классов в моделях классификации. Используется для устранения дисбаланса (например, вес `positive`= `количество_negative`/`количество_positive`);
- `classes_count` - количество классов для многоклассовой классификации;
- `auto_class_weights` - устраняет дисбаланс автоматически (значения - **`Balanced`**, **`SqrtBalanced`**);
- `scale_pos_weight` - задает вес для положительного класса; Среди параметров `scale_pos_weight`, `auto_class_weights`, `class_weights` одновременно используется только один.

##### Best practices по надстройке

- 🍏 Категориальные признаки помещать в `cat_features`. Сами не кодируем.
- ⚠️ Фиксировать `random_seed` модели и разбиения. (В идеале, порядок фичей тоже)
- 🍏 Помещать данные в `Pool` для ускорения. (Квантилизация заранее)
- 🎓 Иметь `eval_set`, чтобы выставить `early_stopping_rounds`
- 🍏 Использовать регуляризационные параметы `l2_leaf_reg`, `colsample_bylevel`, `subsample` и другие
- 🎓 Ставить `max_depth` как можно меньше при том же уровне точности
- 🍏 `iterations` должен быть с запасом для срабатывания `early_stopping_rounds`
- 🎓 `learning_rate` подбирается по метрике.
- 🍏 Метрика - та, что заявлена в чемпионате. (**Recall**, **MAE**, **Корреляция** ...)
- ⚠️ Оцениваем модель по **`score = mean(metrics) - std(metrics)`**!
- 📈 Иногда записываем значения метрики локально и их значение на лидерборде.

#### LightGBM
##### Особенности
Реализация вводит две ключевые идеи: `GOSS` и `EFB`.  
- С помощью `GOSS` (Градиентная односторонняя выборка) мы исключаем значительную часть экземпляров данных с небольшими градиентами и используем оставшиеся экземпляры для оценки прироста информации. Было доказано, что, поскольку экземпляры данных с большими градиентами играют более важную роль, `GOSS` может получить довольно точную оценку с гораздо меньшим размером данных.
- `EFB` или `Exclusive Feature Bundling` (объединение взаимоисключающих признаков) — это подход объединения разреженных (в основном нулевых) взаимоисключающих признаков, таких как категориальные переменные, закодированные onehot-кодированием. Это, по сути, тип автоматического подбора признаков.  Мы пакетируем взаимоисключающие признаки (то есть они редко принимают ненулевые значения одновременно), чтобы уменьшить количество признаков.
##### Parameter notes [Pt1]
**Чтобы повысить точность модели:** 📈🎯

- Использовать большой `max_bin` (замедляет обучение)
- Уменьшить `learning_rate`, увеличив число деревьев `num_iterations`
- Увеличить `num_leaves` (может привести к оверфиттингу)
- Попробовать `dart` 🦄  
**Для борьбы с переобучением:** 🚀🚢
- Уменьшить `max_bin` и `num_leaves`
- Использовать `min_data_in_leaf` и `min_sum_hessian_in_leaf`
- Использовать [бэггинг], указав `bagging_fraction` и `bagging_freq`
- Использовать сабсэмплинг признаков, установив `feature_fraction`
- Попробовать `lambda_l1`, `lambda_l2`, `min_gain_to_split` и `extra_trees` для регуляризации
- Установить `max_depth` для ограничения глубины дерева

##### Parameter notes [Pt2]
**Dataset creation**
- Как в библиотеке `CatBoost` есть класс `Pool` для создания датасетов, так и в `LightGBM` есть класс `Dataset` для той же цели. Как заявлено, является очень эффективным по потребляемой памяти, т.к. хранит не конкретные значения признаков, а только их дискретные бины.
- Число бинов, в свою очередь, можно отрегулировать при создании датасета, уменьшив параметры `max_bin` или `max_bin_by_feature`.
- В документации заявлено, что можно подавать категориальные фичи без one-hot энкодинга. Фреймворк сам обработает, что выйдет быстрее до 8 раз.
- ⚠️ Однако, реализовано это специфическим образом: перед созанием класса `Dataset` категориальные фичи надо привести к целочисленному типу (`int`).
- ⚠️ Целевая переменная - (параметр `label`) также ограничен по возможным типам: `int`, `float` или `bool`
- `categorical_feature`
- `feature_name` — принимает список строк, определяющих имена столбцов.
- `reference` — (`Dataset` или `None` на выбор (`default=None`)) – Если это `Dataset` для валидации, то тренировочный `Dataset` должен быть указан как референсный.
- `group` – используется при решении задачи ранжирования (обучении Ранкера).
- `weight` – можно указать список весов для каждого экземпляра данных (неотрицательные числа) или установить позже, используя метод `set_weight()`.
- `params` – принимает словарь параметров, здесь, как раз, можно указать количество бинов (`max_bin` или `max_bin_by_feature`).
- `free_raw_data` (`bool`, optional (`default=True`)) – отвечает за освобождение памяти от сырых данных после создания датасета.
**Параметры обучения**
- `params` 
- `train_set` — принимает объект типа `Dataset`, который содержит информацию о признаках и целевых значениях.
- `num_boost_round` — указывает количество деревьев бустинга, которые будут использоваться в ансамбле (по умолчанию 100).
- `valid_sets` — принимает список `Dataset` объектов, которые являются выборками для валидации. Эти проверочные датасеты оцениваются после каждого цикла обучения.
- `valid_names` — принимает список строк той же длины, что и у `valid_sets`, определяющих имена для каждой проверочной выборки. 
- `categorical_feature` 
- `verbose_eval` 
- Тип бустинга указывается с помощью аргумента `boosting_type`.
- `objective` — этот параметр позволяет нам определить целевую функцию
- `metric`
- `num_iterations` 
- `learning_rate`   
- `num_leaves` 
- `max_depth` — этот параметр позволяет нам указать максимальную глубину, разрешенную для деревьев в ансамбле. По умолчанию -1, что позволяет деревьям расти, как можно глубже.
- `min_data_in_leaf` — данный параметр принимает целочисленное значение, определяющее минимальное количество точек данных (семплов), которые могут храниться в одном листе дерева. Этот параметр можно использовать для контроля переобучения. Значение по умолчанию 20.  
- `bagging_fraction` — этот параметр принимает значение с плавающей запятой от 0 до 1, которое позволяет указать, насколько большая часть данных будет случайно отбираться при обучении. Этот параметр может помочь предотвратить переобучение. По умолчанию 1.0.  
- `feature_fraction` — данный параметр принимает значение с плавающей запятой от 0 до 1, которое информирует алгоритм о выборе этой доли показателей из общего числа для обучения на каждой итерации. По умолчанию 1.0, поэтому используются все показатели.
- `extra_trees` — этот параметр принимает логические значения, определяющие, следует ли использовать чрезвычайно рандомизированное дерево или нет.
- `force_col_wise` — этот параметр принимает логическое значение, определяющее, следует ли принудительно строить гистограмму по столбцам при обучении. Если в данных слишком много столбцов, установка для этого параметра значения `True` повысит скорость процесса обучения за счет уменьшения использования памяти.  
- `force_row_wise` — этот параметр принимает логическое значение, определяющее, следует ли принудительно строить гистограмму по строкам при обучении. Если в данных слишком много строк, установка для этого параметра значения `True` повысит скорость процесса обучения за счет уменьшения использования памяти. 
- `early_stopping_round` — принимает целое число, указывающее, что мы должны остановить обучение, если оценочная метрика, рассчитанная на последнем проверочном датасете, не улучшается на протяжении определенного параметром числа итераций.
- `num_class` — если мы работаем с задачей мультиклассовой классификации, то этот параметр должен содержать количество классов.  
- `is_unbalance` — это логический параметр, который должен иметь значение `True`, если данные не сбалансированы. Его следует использовать с задачами бинарной и мультиклассовой классификации.
##### Plotting importance
- plot_importance
- plot_split_value_histogram



#### XGBoost
##### Некоторые фишки
- Много параметров регуляризации (`lambda`/`gamma`/`alpha`)
- Считается самым быстрым и менее ресурсозатратным по сравнению с другими бустингами (холиварное утверждение)
- Встроенная работа с пропусками и категориальными признаками (просто `one-hot encoding` ) (`enable_categorical = True` / `df[cat_col].astype('category')`)
- Поддержка метрик из библиотеки **scikit-learn** в качестве кастомных из коробки.

- Поддержка работы с `Dask` и `Spark` из коробки.
- Поддерживает работу с популярными облачными сервисами и распределенное обучение на кластере или нескольких машинах.
- Поддерживает большинство популярных языков программирования
- Поддерживается другими фреймворками и сервисами (**Optuna**, **Weights & Biases** и пр.)

##### Особенности реализации бустинга
- **Параллелизация:** В `XGBoost` построение деревьев основано на параллелизации. Это возможно благодаря взаимозаменяемой природе циклов, используемых для построения базы для обучения: внешний цикл перечисляет листья деревьев, внутренний цикл вычисляет признаки. Нахождение цикла внутри другого мешает параллелизировать алгоритм, так как внешний цикл не может начать своё выполнение, если внутренний ещё не закончил свою работу. Поэтому, для улучшения времени работы, порядок циклов меняется: инициализация проходит при считывании данных, затем выполняется сортировка, использующая параллельные потоки. Эта замена улучшает производительность алгоритма, распределяя вычисления по потокам.
- **Отсечение ветвей дерева** (`gamma`): В фреймворке `GBM` критерий остановки для разбиения дерева зависит от критерия отрицательной потери в точке разбиения. `XGBoost` использует параметр максимальной глубины `max_depth` вместо этого критерия и начинает обратное отсечение. Этот “глубинный” подход значительно улучшает вычислительную производительность.
- **Аппаратная оптимизация:** Алгоритм был разработан таким образом, чтобы он оптимально использовал аппаратные ресурсы. Это достигается путём создания внутренних буферов в каждом потоке для хранения статистики градиента.
- **Регуляризация:** Он штрафует сложные модели, используя как регуляризацию `LASSO` (L1), так и `Ridge`-регуляризацию (L2), для того, чтобы избежать переобучения.
- **Работа с пропусками:** Алгоритм упрощает работу с разреженными данными, в процессе обучения заполняя пропущенные значения в зависимости от значения потерь.
- **Кросс-валидация:** Алгоритм использует свой собственный метод кросс-валидации на каждой итерации.

##### Parameter notes [Pt1]
- booster: gbtree, dart,gblinear
- tree_method: 'hist' or 'gpu_hist' ; 'auto' ; 'exact' for small datasets ; 'approx' for quantiles and gradient histogram for big datasets
- lambda, alpha : L1 and L2 regularization coeffs
- gamma : min loss threshold for branch split

**Тюнинг**
- Первый - это контроль сложности модели, с помощью параметров:
    - `max_depth` - уменьшить
    - `min_child_weight` - увеличить
    - `gamma` и `lambda` - увеличить
- Второй - это добавить случайность в модель, чтобы сделать её устойчивой к шуму:
    - `subsample` - уменьшить
    - `colsample_bytree`, `colsample_bylevel`, `colsample_bynode` - уменьшить
    - использовать бустер `DART`

`dart` - Режим, в котором есть `dropout`, аналогично как в нейронных сетях.
- `rate_drop` (default=0.0) - `Dropout rate` (доля предыдущих деревьев, которые будут отброшены во время `dropout`)
- `one_drop` (default=0) (0 или 1) - когда 1, хотя бы 1 дерево всегда будет отброшено при `dropout`
- `skip_drop` (default=0.0) - вероятность, что `dropout` не будет в течение итерации бустинга (имеет больший приоритет, чем 2 предыдущих параметра)  
    **⚠️ Если ни один из этих параметров не указать (оставить дефолтные значения), то бустер будет работать в режиме** `gbtree`

###### ⚠️ **Важное замечание:** 
Использование `predict()` c `DART` бустером.  
Если объект бустера относится к типу `DART`, функция `predict()` выполнит отсев (`dropout`), т.е. будут использованы только некоторые деревья. Это приведет к неправильным результатам, если данные не являются обучающей выборкой. Чтобы получить правильные результаты на тестовой выборке, установите `iteration_range` в ненулевое значение  

**Например:** `preds = bst.predict(dtest, iteration_range=(0, num_round))`

🍏 **Интересный факт:** В библиотеке `LightGBM` есть параметр `xgboost_dart_mode` (True или False), видимо у них различные реализации `DART`.

##### Parameter notes [Pt2]
**DMatrix parameters**
- `data` - данные для обучения
- `label` - таргет
- `nthread` - количество параллельных потоков при загрузке данных (`-1` означает все потоки)
- `missing` (по умолчанию `np.nan`) - можно указать значение, которым заменить пропуски.
- `enable_categorical` - сам определит категориальные фичи, если они были предварительно переведены в тип `category` для `pandas.DataFrame`.*  
- Как и в `LightGBM`, кат. фичи из строк надо предварительно перевести к целочисленному типу, т.е. провести `label-encoding`.
**Important hyperparameters**
- `booster`
- `tree_method`
- `eta` (default=0.3, aka: `learning_rate`) - шаг обучения
- `max_depth` (default=6) - максимальная глубина дерева. Чем больше глубина, тем сложнее модель и более склонна к оверфиту.
- `num_boosting_rounds` (default=100, в sklearn API `n_estimators`) - количество деревьев(итераций) бустинга.
- `lambda` (default=1, alias: `reg_lambda`) и `alpha` (default=0, alias: `reg_alpha`) - отвечают за L2 и L1 регуляризации соответственно. Чем больше значение параметров, тем более консервативной (менее склонной к переобучению, но может упускать актуальную информацию) становится модель. Рекомендуемые значения находятся в диапазоне `0–1000` для обоих.
- `gamma` (default=0, aka: `min_split_loss`) - значение минимального изменения лосса для разделения ветви дерева - чем больше `gamma`, тем более консервативной будет модель.
- `min_child_weight` (default=1) - Если шаг разделения дерева приводит к листовому узлу с суммой весов экземпляров меньше, чем `min_child_weight`, то процесс построения откажется от дальнейшего разделения. Чем больше `min_child_weight`, тем более консервативной будет модель.
- `subsample` (default=1) - доля экземпляров из обучающей выборки, которая будет использована для построения дерева. 0.5 - берем половину. Обновляется на каждой итерации.
- `colsample_bytree`, `colsample_bylevel`, `colsample_bynode` (default=1) - семейство параметров по отсечению доли признаков на определенном уровне построения дерева.
- `max_bin` (default=256) - максимальное число бинов, на которое может быть разбит числовой признак. Увеличение значения этого параметра может сделать бинаризацию более оптимальной, но увеличит время обучения.
- `objective` (`default=reg:squarederror`) - лосс-функция. **Определяет задачу, которую вы решаете** , `reg:squaredlogerror` - квадратичная ошибка.
- `eval_metric` - функция оценки.

#### Optuna [Tuning]
- Хотя `Random Search` уже значительно ускоряет процесс поиска, мы можем пропустить набор гиперпараметров, при котором модель показывает лучшее качество.  
- И тут, в голову может прийти идея: "А что, если нам немного поугадывать вначале, как в `Random Search`. А потом будем чаще проверять в тех местах, рядом с которыми модель показала лучшую точность?!". Такой метод называется - **Байесовский поиск гиперпараметров модели**.
- Самые популярные библиотеки, в которых реализован этот метод - `HyperOpt` и `Optuna`. (в нашей практике c `HyperOpt` часто случаются сбои и нестабильная работа, поэтому в этом ноутбуке сосредоточимся на **`Optuna`**)
##### Ключевые особенности Optuna
- 🎯 Легковесность и универсальность - можно подбирать оптимальные параметры под любые функции и метрики
- 🎁 SOTA алгоритмы, адаптированные для поиска гиперпараметров
- ⏱ Параллелизация и различные методы прунинга
- 📈 Встроенная визуализация
- 🤝 Интеграция со множеством популярных библиотек (бустинги, **sklearn**, **PyTorch**, **W&B** и другие)
##### Cущности Optuna
В `optuna` присутствуют две базовые сущности - `study` и `trial`.
В первой находится код подсчета метрики, во второй параметры для перебора. К примеру:
```python

param = trial.suggest_float('param', 0, 1.5) 

loss_function = trial.suggest_categorical('loss', ['Logloss', 'CrossEntropy'])

depth = trial.suggest_int('depth', 5, 8)

learning_rate = trial.suggest_uniform('learning_rate', 0.0, 1.0)
```

**Study parameters**
Инициализируем обьект `study`, который начнет перебор и сохранит в себе историю результатов. Если мы стараемся увеличить метрику, а не уменьшить ошибку, то используем `create_study(direction='maximize')`

```
study = optuna.create_study()
study.optimize(objective, n_trials=10)
```
В [`Optuna`](https://optuna.readthedocs.io/en/stable/index.html) реализовано несколько методов (`sampler`) подбора параметров (в том числе классические):
- `GridSampler`
- `RandomSampler`
- `Tree-Structed Parzen Estimator` (`TPESampler` - самый популярный - дефолтный)
- `BruteForceSampler`
- И ещё [4 других](https://optuna.readthedocs.io/en/stable/reference/samplers/index.html#module-optuna.samplers), также можно написать собственный сэмплер.
**Пример**
```
# Ограничим логирование
optuna.logging.set_verbosity(30)

def objective(trial):
    x = trial.suggest_float("x", -8, 10)
    return (x + 1) * (x + 5) * (x - 9)

# создадим объект обучения, и запустим на 10 итераций; т.к мы ищем минимум, параметр direction оставляем дефолтным
study = optuna.create_study()

# запуск поиска
study.optimize(objective,
               n_jobs=-1,
               n_trials=250,
               show_progress_bar=True)
```

##### Советы по перебору параметров (очень логичные)
- Иметь понимание важности параметров
- Число `iterations` лучше взять с запасом и зафиксировать, при этом ограничив через `early_stopping_rounds`
- Подсмотреть/чувствовать диапазоны и шаг значений
- Исключить то, что перебирать не нужно. (`random_seed` , `eval_metric`, `thread_count` и прочее)
- Используйте информацию с прошлых попыток
###### fit catboost
```
def fit_catboost(trial, train, val):
    X_train, y_train = train
    X_val, y_val = val
    param = {
        'iterations' : 400, # Можно не перебирать, есть Early-Stopping
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.01),
        "l2_leaf_reg": trial.suggest_int("l2_leaf_reg", 2, 50),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.8),   
        "auto_class_weights": trial.suggest_categorical("auto_class_weights", ["SqrtBalanced", "Balanced", "None"]),
        "depth": trial.suggest_int("depth", 3, 9),
        "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
        "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
        "used_ram_limit": "14gb",
        "eval_metric": "Accuracy", # Тоже стоит заранее определиться
    }

    
    if param["bootstrap_type"] == "Bayesian":
        param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 20)
        
    elif param["bootstrap_type"] == "Bernoulli":
        param["subsample"] = trial.suggest_float("subsample", 0.1, 1)
    clf = CatBoostClassifier(
        **param,
        thread_count=-1,
        random_seed=42,
        cat_features=cat_features,
    )
    clf.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        verbose=0,
        plot=False,
        early_stopping_rounds=5,
    )

    y_pred = clf.predict(X_val)
    return clf, y_pred
```
###### objective function
```
def objective(trial, return_models=False):
    n_splits = 3
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    X_train = train[filtered_features].drop(targets, axis=1, errors="ignore")
    y_train = train["target_class"]
    scores, models = [], []
    for train_idx, valid_idx in kf.split(X_train):
        train_data = X_train.iloc[train_idx, :], y_train.iloc[train_idx]
        valid_data = X_train.iloc[valid_idx, :], y_train.iloc[valid_idx]
        # Подаем trials для перебора
        model, y_pred = fit_catboost(trial, train_data, valid_data) # Определили выше
        scores.append(accuracy_score(y_pred, valid_data[1]))
        models.append(model)
        breal     
    result = np.mean(scores)
    if return_models:
        return result, models
    else:
        return result
```
###### Запускаем обучение
```
study = optuna.create_study(direction="maximize")
study.optimize(objective,
               n_trials=600,
               n_jobs = -1,
               show_progress_bar=True,)
print("Best trial: score {}, params {}".format(study.best_trial.value, study.best_trial.params))
```
Обучим итоговую модель на лучших параметрах:
```
valid_scores, models = objective(
    optuna.trial.FixedTrial(study.best_params),
    return_models=True,
)
```

###### Визуализация
```
trials_df = study.trials_dataframe().sort_values('value', ascending=False)
trials_df.head(3)

# История изменения от числа испытаний
optuna.visualization.plot_optimization_history(study)

# Зависимость в разрере по параметрам
params = ['l2_leaf_reg', 'colsample_bylevel', 'bagging_temperature', 'depth', 'bootstrap_type', 'subsample']
optuna.visualization.plot_slice(study,
                                params=params,
                                target_name = 'accuracy_score')
# Важность параметров
optuna.visualization.plot_param_importances(study)
```
###### Pruning
Реализации прунеров есть для большинства известных ML фреймворков. Например, `callbacks` для бустингов:

- `CatBoost` : `optuna.integration.CatBoostPruningCallback`
- `XGBoost` : `optuna.integration.XGBoostPruningCallback`
- `LightGBM` : `optuna.integration.LightGBMPruningCallback`

**Основные виды прунеров**:
- `Median Pruner` - самый популярный, каждые несколько итераций отбрасывает половину процессов с наихудшим качеством
- `Successive Halving Algorithm` (SHA) - сначала запускаются `trials` с минимальными ресурсами (мало обучающих примеров, мало итераций), и на каждом следующем шаге отсекаем половину `trials` с худшим качеством + увеличиваем ресурсы
- `Percentile Pruner`, `Hyperband Pruner` и [другие](https://optuna.readthedocs.io/en/stable/reference/pruners.html)

```
Внутри objective_catboost

from optuna.integration import CatBoostPruningCallback
pruning_callback = CatBoostPruningCallback(trial, "Accuracy")

clf.fit(
        X_train,
        y_train,
        eval_set=(X_test, y_test),
        verbose=0,
        plot=False,
        early_stopping_rounds=5,
        callbacks=[pruning_callback],
    )  # Добавляем callback в fit

# запускаем процесс прунинга
pruning_callback.check_pruned()
```
###### Примечание
- 🍏 У `Optuna` ещё множество полезных применений, вплоть до [препроцессинга датасетов](https://t.me/ds_private_sharing/60), ограничения только в вашей фантазии.
- 🍏 [Официальные туториалы](https://optuna.readthedocs.io/en/stable/tutorial/index.html) с примерами.