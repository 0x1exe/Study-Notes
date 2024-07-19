# American express - default prediction
#### Датасет в 50 гигов данных, а хватит ли оперативки открыть?
**Решение**: заранее подаем в Pandas словарь типов, с которыми надо импортировать столбцы.![[Large DF.png]]
По умолчанию Pandas подгружает данные в самом "тяжелом" формате - `float64`. Поэтому банальная смена типа на `float16` уменьшала размер датасета в 10 раз
##### Code to reduce memory usage: 
```python
def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    # print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
          col_type = df[col].dtype.name

          if col_type not in ['object', 'category', 'datetime64[ns, UTC]']:
              c_min = df[col].min()
              c_max = df[col].max()
              if str(col_type)[:3] == 'int':
                  if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                      df[col] = df[col].astype(np.int16)
                  elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                      df[col] = df[col].astype(np.int16)
                  elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                      df[col] = df[col].astype(np.int32)
                  elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                      df[col] = df[col].astype(np.int64)
              else:
                  if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                      df[col] = df[col].astype(np.float16)
                  elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                      df[col] = df[col].astype(np.float32)
                  else:
                      df[col] = df[col].astype(np.float64)

  # end_mem = df.memory_usage().sum() / 1024**2
  # print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
  # print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

  return df = reduce_mem_usage(df)
```
#### Извлекаем ембеддинги и укращаем pytorch-lifestream
В качестве основы для ембеддингов решено было попробовать библиотеку **pytorch**-**lifestream** от разработчиков из Сбера, которая хорошо себя зарекомендовала (1 и 3 места), на проходившем недавно Data Fusion Contest, тоже на данных банковских транзакций

Так же попытались обучить автоэнкодер и трансформер из этой же библиотеки, но не удалось выбить из них большой скор (0.789 и 0.791). Однако, мы использовали их предсказания в итоговом ансамбле, что увеличило устойчивость модели.
#### Извлекаем ембеддинги и укрощаем tsfresh:
Несмотря на то, что [tsfresh](https://tsfresh.readthedocs.io/en/latest/) достаточно медленно генерирует признаки, зато это сразу и разнообразный и достаточно боевой набор признаков. Но вот ведь незадача, если запустить генерацию на всех сырых данных, то есть два исхода. Либо генерация будет длится несколько суток, либо в один момент объем накопленных признаков привысит лимит по памяти. И даже если распраллелить на на несколько CPU, все равно придется ждать около 4 дней.

**Решение**: посчитать список признаков на 10% данных, далее отфильтровать топ-2000 самых релевантных. Так и сделали. Как итог, распараллелив процесс на 4 CPU мы ускорили генерацию до нескольких дней, получая на выходе полный набор релевантных признаков. Для большей стабильности генерации признаков - сохраняли промежуточные результаты на каждой итерации. Вышло примерно 4К фичей, а это все еще много, но уже терпимо

**Замечание**: Более умным путем было бы использовать вместо **tsfresh** ускоренный аналог - **tsfel**. Однако, у нас на сервере **tsfel** запускаться отказывался =(.

###### Code for feature generation with tsfresh
``` python
# settings = get_setting_tsfresh(settings)
# settings = MinimalFCParameters()
settings = EfficientFCParameters()

train_df.fillna(0, inplace=True)
res_train_df = None
# res_test_df = None

vr = VarianceThreshold(0.5)
for num, num_col in enumerate(numeric_cols):
    now = datetime.now()
    print(num, 'col', num_col)
    
    
    settings = full_settings_filtered[num_col]
    
    Distributor = MultiprocessingDistributor(n_workers=4,
                                             disable_progressbar=False,
                                             progressbar_title="Feature Extraction")

    X = extract_relevant_features(train_df[["customer_ID", "S_2"]+[num_col]].fillna(0), 
                                  y,
                                  column_id='customer_ID',
                                  column_sort='S_2',
                                  n_jobs=5,
                                  chunksize=5,
                                  default_fc_parameters=settings,
                                  fdr_level = 0.01,
                                  distributor = Distributor)
    
    X = pd.DataFrame(vr.fit_transform(X), columns=X.columns[vr.get_support()])
    print('прошло времени до генерации', datetime.now() - now)
    kind_to_fc_parameters = tsfresh.feature_extraction.settings.from_columns(X)
    X_test = extract_features(test_df[["customer_ID", "S_2"]+[num_col]].fillna(0), 
                                  column_id='customer_ID',
                                  column_sort='S_2',
                                  n_jobs = 4,
                                  chunksize=5,
                                  distributor = Distributor,
                                  kind_to_fc_parameters=kind_to_fc_parameters)
    X_test = X_test[X.columns]
    print('прошло времени до фильтрации', datetime.now() - now)
    if res_train_df is None:
        res_train_df = X
        res_train_df["target"] = y.values
        res_train_df["customer_ID"] = customer_train
        res_test_df = X_test
        res_test_df["customer_ID"] = customer_test
    else:
        X = reduce_mem_usage(X)
        X_test = reduce_mem_usage(X_test)

        res_train_df = pd.concat([res_train_df, X], axis=1)
        res_test_df = pd.concat([res_test_df, X_test], axis=1)

        res_train_df.to_csv("./../tmp_data/del_full_train_tsfresh.csv", index=False)
        res_test_df.to_csv("./../tmp_data/del_full_test_tsfresh.csv", index=False)
        
        print('прошло времени до сохраниения', datetime.now() - now)
```


#### 💧 Фильтрация признаков или как убрать 50% мусора?!
| **Метод** | **Преимущество** | **Проблема/Преимущество** |
| ---- | ---- | ---- |
| Feature Importance | Быстрый расчет | Неточный. Отсеивание "неважных" признаков не реже помогает поднять скор |
| Permutation Importance,  <br>Target Importance, Shap | Очень долгий расчет | Как правило, помогает убрать мусорные фичи |
| Корреляции, Критерии | Быстрый расчет | Неточный. Не ловит сложные зависимости. |
| Рекурсивное удаление | Долго | Точно |
Вообще говоря, методов фильтации признаков много, но все упиралось во время. На этап фильтрации признаков и борьбу за последние доли точности у нас уже не оставалось достаточного колличества времени.

Помогло три трюка. **Первое** - взять для фильтрации признаков 10% данных, **второе** - отсеивать признаки не по одному, а группами. **Третье** - если удаление группы признаков поднимало скор на лидерборде (не локально) - то мы выбрасывали все признаки этой группы без исключения. Так всего через сутки мы смогли ужать 4 тысячи признаков до 2 тысяч. При этом новый новый скор поднял нас всего лишь на несколько десятков мест вверх на бронзовый порог. (Напомню, что всего в чемпионате было почти 5 000 участников).

#### 😈 Модели - смешать, но не взбалтывать!
![[final_pipeline.png]]
Основной частью решения тут является смесь градиентных бустингов **Catboost** и **LightGBM**. Только бустингов для конкурентноспособного скора не хватало. Однако, смешивание c диверсифицированной моделью **RNN Transformer**'а добавило нам в точности. Замечу, что **lightGBM** тут работал в режиме dart (это такой режим, где есть **dropout**'ы по аналогии с нейронками)

Попробуем разобраться в чем секрет такого шейк-апа на 700 мест вверх? Есть три причины.

- Как видно из решения - финальная модель состояла из ансабля уже нескольких сильных моделей, что хорошо стабилизирует модели.
- Обучение бустингов происходило по фолдам, т.е. финальный прогноз стабилизировался несколькими моделями. Так можно использовать все данные при обучении, и это не мешает валидации.
- Важная фишка - это декомпозиция модели по пользователям, у которых представлены не все признаки. Дело в том, что 10% пользователей в датасете имели не полный набор данных (всего пару месяцев вместо 12). Логично было при обучении натравить на такие случаи отдельную урезанную модель на основе 500 признаков. Далее, для таких пользователей ответы моделей не смешивались. Бралась только одна из моделей в зависимости от числа имеющегося числа признаков.

#### ⚙️Что могло технически помочь, но мы не попробовали

- Библиотека для работы с датафреймами (и не только) на GPU - **rapids** ([rapids.ai](http://rapids.ai/))
    
- **cudf** - это эквивалент Pandas для GPU
    
- **FIL** - библиотека для инференса моделей из `sklearn`, бустингов типо XGBoost / LightGBM на GPU с кучкой «хаков» для ускорения.
    
- Умение параллелить процессы на CPU ( `n_jobs` / `n_threads` )
    
    Все это продукты Nvidia - не просто сторонний софт
