## Plotting

- What do we need:
###### [Code] Plot dataframe numerical features
```python
def plot_dataframes_features_info(df1, df2, name):

    print('\n', name, '\n')

    for column in df1.describe().columns:

        plot_numerical_feature_histogram(df1, df2, column, figure_width=10, figure_height=5)
```
```
def plot_numerical_feature_histogram(df1, df2, feature_title, label=None, text_info=True, bins=50,
                                     figure_width=17, figure_height=7, fontsize=15, number_of_decimals=2, title=''):
    NCOLS_NUMBER_CONST = 1
    if label is None:
        label = feature_title
    fig, ax = plt.subplots(ncols=1, figsize=(figure_width, figure_height))
    sns.histplot(data=df1, x=feature_title, color="cornflowerblue", label=label, bins=bins, kde=True)
    mean_value = df1[feature_title].mean()
    median_value = df1[feature_title].median()
    ax.axvline(x=mean_value, color='black', lw=2, ls=':',
               label=f"mean: {round(mean_value, number_of_decimals)}")
    ax.axvline(x=median_value, color='orange', lw=2, ls='-.',
               label=f"median: {round(median_value, number_of_decimals)}")
    ax.legend(fontsize='15', title_fontsize="10", loc='upper right')
    plt.tight_layout()

    Q1_QUANTILE_VALUE_CONST = float(0.25)
    Q3_QUANTILE_VALUE_CONST = float(0.75)
    IQR_MULT_DEFAULT_CONST = float(1.5)
    Q1 = df1[feature_title].quantile(Q1_QUANTILE_VALUE_CONST)
    Q3 = df1[feature_title].quantile(Q3_QUANTILE_VALUE_CONST)
    IQR = Q3 - Q1

    sns.histplot(data=df2, x=feature_title, color="orange", label=label, bins=bins, kde=True)
    mean_value = df2[feature_title].mean()
    median_value = df2[feature_title].median()
    ax.axvline(x=mean_value, color='green', lw=2, ls=':',
               label=f"mean: {round(mean_value, number_of_decimals)}")
    ax.axvline(x=median_value, color='red', lw=2, ls='-.',
               label=f"median: {round(median_value, number_of_decimals)}")
    ax.legend(fontsize='15', title_fontsize="10", loc='upper right')
    plt.tight_layout()  

    Q1_QUANTILE_VALUE_CONST = float(0.25)
    Q3_QUANTILE_VALUE_CONST = float(0.75)
    IQR_MULT_DEFAULT_CONST = float(1.5)
    Q1 = df2[feature_title].quantile(Q1_QUANTILE_VALUE_CONST)
    Q3 = df2[feature_title].quantile(Q3_QUANTILE_VALUE_CONST)
    IQR = Q3 - Q1
    plt.show()
```
###### trends
- lineplot

###### distribution
- distplot
- jointplot
- kdeplot

###### relations
- barplot
- heatmap
- scatter plot
- swarm
- regplot
- lmplot


## Error visualization
###### Fold feature importance
- Compute mean across folds
- box plot

###### SHAP tools
- TreeExplainer
- dependence_plot
- waterfall

###### Classification report: confusion_matrix

###### PartialPlots

###### Plotting residuals
- Plot residuals
- Plot residuals distribution (should be normal)
- Residuals vs. actual plotting
- Actual data vs predictions or (True - pred) vs. (True + pred)


## Feature engineering techniques
- imputing missing data
- dealing with outliers 
- binning
- log transformation 
- data scaling
- one-hot encoding 
- handling categorical and numerical variables 
- creating polynomial features 
- dealing with geographical data
- working with date data
##### Tools
- [Pyfeat]
- [Cognito]
- [Tsfresh]
- [Autofeat]
- [FeatureTools]

##### Best practices
###### Indicator variables
Use indicator variables to incorporate vital information in the model. Here is how you would go about this:
- **creating indicator variables from thresholds.** Let’s go back to that car insurance example again. For instance, let’s say that most accidents happen on the 20th day of the month. You can, for example, create a variable that indicates if the day of the month is greater than or equal to 20. 
- **indicator variables for special events**. In the case of car accidents, you can create flags to inform the algorithm if there are significant beer sale events at specific points of the year. It could be that there are more accidents and hence more insurance claims during these periods. 
- **create indicator variables from multiple features.** For example, if people who are alcoholics and are from a particular location make more claims, you can create an indicator variable for that. 
- **indicator variable for group classes**. In the data, you might have a field indicating how the client became a client of the insurance company. Some of these sources could be radio advertising, tv advertising, walk-in, website, etc. You may want to create a new flag that indicates if the client joined from a paid or a free source.
###### Interaction features
Other times you can create features from the interaction of two or more features. Here are some ways this can be applied:
- the quotient of two features
- the difference of two features
- product of two features
- the sum of two features


##### Imputting missing data
The solution is to fill the missing values with statistical estimates. For example, the missing values can be filled with the:
- mean 
- mode 
- median
```
mean_us = usa['loan_amount'].mean()
usa['loan_amount'] = usa['loan_amount'].fillna(median_us)
```

The same can be achieved using [Scikit-learn](https://scikit-learn.org/). The `SimpleImputer` [function](https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html) can be used for this
##### Categorical values handling
- **ordinal**: these categories are ordered. For instance, high is bigger than medium and low when considering salary ranges, i.e., high > medium > low. The inverse is also true
- **non-ordinal**: these categories have no order. For example, in the case of sectors, agriculture is not higher than housing.

One popular tool for doing this is  [category_encoders]
```python
import category_encoders as ce
encoder = ce.OrdinalEncoder(cols=cat_feats)
encoded_data = encoder.fit_transform(usa)

### One-hot encoding
one_hot = ce.OneHotEncoder(cols=cat_feats)
oe_data = one_hot.fit_transform(usa)
```
Hash encoding can also be tried. It is a multivariate hashing implementation with configurable dimensionality. It doesn’t maintain a dictionary representation of the categories. It, therefore, doesn’t grow in size:
```python
hash_enc = ce.HashingEncoder(cols=cat_feats, n_components=8)
hash_enc_data = hash_enc.fit_transform(usa)
```
##### Numerical and continuous features

1.Some common strategies for normalizing data are using the:
- **standard scaler**: standardizes features by removing the mean and scaling to unit-variance
- **min max scaler**: transforms the numerical values by scaling each feature to a given range; for example forcing all values to be between zero and one
- **robust scaler**: scales the numerical values using statistics that are robust to outliers. It gets rid of the median and scales the data according to the quantile range
```python
rsc = RobustScaler()
scaled_data = rsc.fit_transform(usa[feat_names])
scaled_data
```
2.Converting numerical values into discrete
```python
usa['loan_group'] = pd.cut(usa['loan_amount'], bins=3, labels=["Low", "Mid", "High"])
```
3.You can use log transformation to center data if it is skewed. When interpreting the results, you have to remember to take the exponent.
```python
usa['loan_amount'] =  np.log1p(usa['loan_amount'])
```

##### Polynomial features

Polynomial features are created by crossing two or more features. This creates a relationship between the independent variables. The relations could result in a model with less bias.

```python
from sklearn.preprocessing import PolynomialFeatures

poly_feats = PolynomialFeatures()
columns_to_cross = ['loan_amount', 'term_in_months', 'lender_count']
crossed_features = poly_feats.fit_transform(usa[columns_to_cross].values)

crossed_features = pd.DataFrame(crossed_features)
```
##### Grouping operations
- These include the likes of: apply, transform, aggregate, groupBy, pivotTable
```python
usa.pivot_table(index='day', 
				columns=['sector'], 
				values='funded_amount', 
				aggfunc=np.sum, 
				fill_value = 0)
```

## Auto feature generation
### Feature tools
Для начала нам нужно создать `EntitySet`. В нём будут наши таблицы и связи между ними.
```
es = ft.EntitySet(id="car_data")
```
Добавим в него таблицы.

В таблицах есть категориальные столбцы `model`, `fuel_tupe`, `car_type`, `years_to_work` и пр. Нам бы не хотелось их суммировать с чем-то или искать среднее, поэтому укажем для `feature-tools` типы данных с помощью библиотеки `woodwork` (устанавливается вместе с `feature-tools`).
```
ft.list_logical_types()
```

```
from woodwork.logical_types import Categorical, Double, Datetime, Age

  

es = es.add_dataframe(
    dataframe_name="cars",
    dataframe=car_info,
    index="car_id",
    logical_types={"car_type": Categorical, 'fuel_type': Categorical, 'model': Categorical}
    )

  

es = es.add_dataframe(
    dataframe_name="rides",
  dataframe=rides_info.drop(['ride_id'], axis=1),
    index='index',
    time_index="ride_date",
    )

  

es = es.add_dataframe(
    dataframe_name="drivers",
    dataframe=driver_info,
    index="user_id",
    logical_types={"sex": Categorical, "first_ride_date": Datetime, "age": Age}
    )

  

es = es.add_dataframe(
    dataframe_name="fixes",
    dataframe=fix_info,
    index="index",
    logical_types={"work_type": Categorical, "worker_id":Categorical}
    )

es
```
Теперь добавим связи между фреймами.
```
es = es.add_relationship("cars", "car_id", "rides", "car_id")

es = es.add_relationship("drivers", "user_id", "rides", "user_id")

es = es.add_relationship("cars", "car_id", "fixes", "car_id")
```
Сгенерируем фичи для машин.
```
feature_matrix, feature_defs = ft.dfs(
    entityset=es,
    target_dataframe_name="cars",
    max_depth=1,
)
feature_matrix.head()
```
Также можно генерировать не всё фичи, а только нужные.
```
feature_matrix, feature_defs = ft.dfs(
    entityset=es,
    target_dataframe_name="cars",
    agg_primitives=["mode", "count"],
    max_depth=1, 
)
feature_matrix.head()
```
Также для фичей можно задавать глубину параметром `max_depth`, тогда фичи будут создаваться не только в пределах таблицы, а ещё и комбинируя признаки со связанных таблиц. А для отладки вычислений можно считать не на всем датасете, что может быть долго, а на нескольких примерах, список которых можно передать в параметр `instance_ids`.
Все доступные типы фичей можно посмотреть в `list_primitives`.
```
ft.list_primitives().head()
```

Так же в `feature-tools` реализован собственный механизм отбора фичей, который имеет 3 функции:

- `ft.selection.remove_highly_null_features()` - отбрасывает признаки с большим количеством пропусков
- `ft.selection.remove_single_value_features()` - отбрасывает константные признаки
- `ft.selection.remove_highly_correlated_features()` - отбрасывает сильно скоррелированные признаки  

В аргументы функции передаем датафрейм и она отрабатывает соответственно названию.
В библиотеке реализовано ещё много полезных фишек, подробнее можно ознакомиться в [документации](https://colab.research.google.com/corgiredirector?site=https%3A%2F%2Ffeaturetools.alteryx.com%2Fen%2Fstable%2Findex.html).

### GeoPandas
Cоздадим GeoDataFrame
```
gdf = gpd.GeoDataFrame(
        df, 
     geometry=gpd.points_from_xy(df['Longitude'], df['Latitude']),
        crs=4326
    ).to_crs(epsg=3857)
gdf.head(3)
```
 Он особо ничем не отличается от обычного DataFrame, единственное отличие в столбце geometry, который представляет собой объект GeoSeries, для которого дополнительно доступны свои атрибуты и методы.
### NetworkX 
```
import networkx as nx
```
Cоздадим две вершины и ребро
```
g = nx.Graph()
g.add_node('П')
g.add_node('С')
g.add_edge('П','С')
nx.draw(g, with_labels=True)
```
С помощью методов: `add_nodes_from()` и `add_edges_from()`, в которые можно передавать элементы списком, можно добавить сразу несколько вершин и ребер
```
g.add_nodes_from(['В','Л'])
g.add_edges_from([('В','П'),('В','С'),('Л','П'),('Л','С')])
nx.draw(g, with_labels=True)
```
Удобно добавить атрибуты к вершинам можно с помощью функции `set_node_attributes`. Она принимает на вход словарь словарей и транспонированный DataFrame в качестве такого сгодится.
```
feat = {'П':{'sum':2000000,'count':10},
        'С':{'sum':1000000,'count':4},
        'В':{'sum':40000,'count':1},
        'Л':{'sum':0,'count':0},}
feat = pd.DataFrame(feat).T
feat

nx.set_node_attributes(g, feat.T)
g.nodes.data()
```
Более сложные фичи из `NetworkX`:
-  Центральность (centrality) - степень важности вершины:
	- `nx.degree_centrality(G)` - чем больше соседей, тем важнее
	- `nx.betweenness_centrality(G)` - cтепень посредничества
	- `nx.closeness_centrality(G)` - степень близости
```
degree_centrality = nx.degree_centrality(G)
degree_centrality_counts = Counter(degree_centrality)
for k, v in degree_centrality_counts.most_common(10):
    print(G.nodes[k]['name'], v)
```
Так, что такое граф разобрались, как фичи добавлять знаем, а как это использовать в ML?

Есть два способа, графовые алгоритмы и графовые нейронные сети. Здесь мы рассмотрим первый способ, а поможет нам с этим швейцарский нож для графов: библиотека `karate-club.`
```
from karateclub import DeepWalk
model = DeepWalk()
model.fit(G)
embedding = model.get_embedding()
print(embedding.shape)
embedding
```
Обученная модель имеет 3 метода:
- `get_embedding` - возвращает ембеддинги узлов
- `get_memberships` - возвращает словарь с принадлежностью узла к тому или иному сообществу; или None
- `get_cluster_centers` - возвращает словарь с информацией является ли узел центром какого-либо сообщества; или None
#### Node2Vec
Теперь разберём более сложный алгоритм `Node2Vec`.

Как вы наверное уже догадались, он тоже преобразует вершины в эмбендинги. Как он это делает? Как и большинство графовых алгоритмов, `Node2Vec` получает структурную информацию из случайных блужданий по графу. Ну или не совсем случайных, поскольку вероятность перехода на новую вершину и уже посещённую регулируемы.` При блужданиях создаётся последовательность из вершин, на которой уже модель Skip-Gram обучается угадывать пропуски в вершинах, так же как в тексте word2vec пытается угадывать пропуски слов.`

```
from karateclub import Node2Vec
n2v = Node2Vec(dimensions=64)

n2v.fit(G)
embeddings = n2v.get_embedding()
embeddings.shape
```
Хоть `Node2Vec` и неплохой алгоритм, но он появился аж в 16-ом году, а в `karate-club` есть алгоритмы и посвежее, так что, если `Node2Vec` не поднимает вас на LB, пробуйте новое, не бойтесь экспериментировать.
#### Пример применения
Представим, у нас задача кредитного скоринга.
Есть Петя, у Пети плохая кредитная история, было много задолженностей. Одобрим ли мы кредит Пете? Скорее всего нет.
А ещё у Пети есть жена Света, она тоже клиент нашего банка. Хорошая КИ, брала несколько крупных кредитов, всё вернула, задолженностей не было.
Мы с вами понимаем, что выплачивать кредит они будут вместе, а значит Пете скор можно и поднять, но на сколько? И какой алгоритм для этого применить?
В принципе, тут можно было бы взять деревья. Но а что если мы захотим использовать информацию о других родственниках? У нас не будет информации по всей родне каждого человека. Или у нас вообще не скоринг, а рекомендательная система в соц. сети и мы хотим предлагать рекламу на основании интересов друзей?
Для таких задач нам нужно работать с графами

### TSFresh
**_TSfresh_** - библиотека, которая автоматически считает большое количество признаков для временных рядов. Так же содержит инструменты для интерпретации силы и важности этих признаков в задачах регрессии и классификации. Теперь сгенерируем фичи для временных рядов.

**_TSFresh_** автоматически извлекает более 100 паттернов из временного ряда. Эти характеристики описывают основные факторы, такие как количество пиков и средние или максимальные значения, а также более сложные факторы, такие как симметричное распределение.

В **_TSfresh_** генерация признаков происходит на основе словаря, который передается в параметр `default_fc_parameters`. Какой словарь туда передадим столько признаков по каждому ряду будет сгенерировано. Словарь состоит из т.н. "калькуляторов фичей". Но в параметр передается не сам словарь, а объект, который возвращает словарь при вызове. Так же стоит отметить, что некоторые вычислители имеют дополнительные аттрибуты и на выходе может получится более 100 признаков.  
Посмотрим на список таких объектов:
- `ComprehensiveFCParameters` (по умолчанию) - полный набор вычислителей - 75 штук (фичей более 100).
- `EfficientFCParameters` - все вычислители кроме самых вычислительно затратных 73 штуки :)
- `MinimalFCParameters` - минимальный набор базовых вычислителей 10 штук. 
Так же можно добавлять или удалять вычислители из этих словарей или добавить свои собственные кастомные вычислители.  Полное описание всех возможных вычислителей фичей смотрите по [ссылке](https://colab.research.google.com/corgiredirector?site=https%3A%2F%2Ftsfresh.readthedocs.io%2Fen%2Flatest%2Ftext%2Flist_of_features.html)

```
from tsfresh.feature_extraction import MinimalFCParameters
from pprint import pprint

pprint(MinimalFCParameters())

from tsfresh import extract_features

extracted_features = extract_features(timeseries, column_id="id", column_sort="time", n_jobs=2, default_fc_parameters=MinimalFCParameters())

extracted_features.head()
```
Cловарь фичь можно редактировать убирая или добавляя иные:
```
fc_parameters = MinimalFCParameters()

for x in ['sum_values', 'median',  'mean', 'maximum', 'absolute_maximum', 'minimum']:
    del fc_parameters[x]

  

fc_parameters.update({
     'linear_trend': [{'attr': 'pvalue'}, {'attr': 'slope'}], 'variance_larger_than_standard_deviation': None,
     'large_standard_deviation': [{'r': 0.05}, {'r': 0.1}]
})

pprint(fc_parameters)

extracted_features = extract_features(timeseries, column_id="id", column_sort="time", n_jobs=2, default_fc_parameters=fc_parameters)

extracted_features.head()
```

Также вы можете указывать какие фичи рассчитывать для каждого отдельного временного ряда.
```
ts_fc_parameters = {
    "F_x": {"mean": None},
    "F_y": {"maximum": None, "minimum": None}
}
```
Ещё одна классная функция в **TSfresh** - `extract_relevant_features` - в неё следует передать не только временные ряды, но и таргет. И фрэймворк сам подберет необходимые признаки, которые лучше сгенерировать для предсказания таргета.
```
from tsfresh import extract_relevant_features

extracted_features = extract_relevant_features(timeseries, y, column_id="id", column_sort="time", n_jobs=4)
extracted_features.head()
```

