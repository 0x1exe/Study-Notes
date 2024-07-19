
## Bootstrapping
The bootstrap method is a resampling technique used to estimate statistics on a population by sampling a dataset with replacement.

### Intro
**The process for building one sample can be summarized as follows**:
1. Choose the size of the sample.
2. While the size of the sample is less than the chosen size
    1. Randomly select an observation from the dataset
    2. Add it to the sample

**The bootstrap method can be used to estimate a quantity of a population**. This is done by repeatedly taking small samples, calculating the statistic, and taking the average of the calculated statistics. We can summarize this procedure as follows:

1. Choose a number of bootstrap samples to perform
2. Choose a sample size
3. For each bootstrap sample
    1. Draw a sample with replacement with the chosen size
    2. Calculate the statistic on the sample
4. Calculate the mean of the calculated sample statistics.

**The procedure can also be used to estimate the skill of a machine learning model.**
1. Choose a number of bootstrap samples to perform
2. Choose a sample size
3. For each bootstrap sample
    1. Draw a sample with replacement with the chosen size
    2. Fit a model on the data sample
    3. Estimate the skill of the model on the out-of-bag sample.
4. Calculate the mean of the sample of model skill estimates.

**Importantly, any data preparation prior to fitting the model or tuning of the hyperparameter of the model must occur within the for-loop on the data sample. This is to avoid data leakage where knowledge of the test dataset is used to improve the model. This, in turn, can result in an optimistic estimate of the model skill.**

A useful feature of the bootstrap method is that the resulting sample of estimations often forms a Gaussian distribution. In additional to summarizing this distribution with a central tendency, measures of variance can be given, such as standard deviation and standard error. Further, a confidence interval can be calculated and used to bound the presented estimate. This is useful when presenting the estimated skill of a machine learning model.


The [resample() scikit-learn function](http://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html) can be used. It takes as arguments the data array, whether or not to sample with replacement, the size of the sample, and the seed for the pseudorandom number generator used prior to the sampling.

For example, we can create a bootstrap that creates a sample with replacement with 4 observations and uses a value of 1 for the pseudorandom number generator.
```python
boot = resample(data, replace=True, n_samples=4, random_state=1)
```
### Application
```
# train model
reg = LogisticRegression(random_state=0)
reg.fit(x_train, y_train)

# bootstrap predictions
accuracy = []
n_iterations = 1000
for i in range(n_iterations):
    X_bs, y_bs = resample(x_train, y_train, replace=True)
    # make predictions
    y_hat = reg.predict(X_bs)
    # evaluate model
    score = accuracy_score(y_bs, y_hat)
    accuracy.append(score)
```
Let’s plot a distribution of accuracy values computed on the bootstrap samples.

```
import seaborn as sns
# plot distribution of accuracy
sns.kdeplot(accuracy)
plt.title("Accuracy across 1000 bootstrap samples of the held-out test set")
plt.xlabel("Accuracy")
plt.show()
```
We can now take the mean accuracy across the bootstrap samples, and compute confidence intervals. There are several different approaches to computing the confidence interval. We will use the percentile method, a simpler approach that does not require our sampling distribution to be normally distributed.
```
# get median
median = np.percentile(accuracy, 50)

# get 95% interval
alpha = 100-95
lower_ci = np.percentile(accuracy, alpha/2)
upper_ci = np.percentile(accuracy, 100-alpha/2)

print(f"Model accuracy is reported on the test set. 1000 bootstrapped samples " 
      f"were used to calculate 95% confidence intervals.\n"
      f"Median accuracy is {median:.2f} with a 95% a confidence "
      f"interval of [{lower_ci:.2f},{upper_ci:.2f}].")
```
**Output**
```
Model accuracy is reported on the test set. 1000 bootstrapped samples were used to calculate 95% confidence intervals.
Median accuracy is 0.86 with a 95% a confidence interval of [0.80,0.91].
```
The confidence interval tells us about the reliability of the estimation procedure. 95% of confidence intervals computed at the 95% confidence level contain the true value of the parameter.

### ⚠️Key points
- Bootstrapping is a resampling technique, sometimes confused with cross-validation.
- Bootstrapping allows us to generate a distribution of estimates, rather than a single point estimate.
- Bootstrapping allows us to estimate uncertainty, allowing computation of confidence intervals.

## Bagging
Bagging, which is short for _bootstrap aggregating_, builds off of bootstrapping. Bootstrap aggregating describes the process by which multiple models of the same learning algorithm are trained with bootstrapped samples of the original data set.

**Ensemble machine learning can be mainly categorized into bagging and boosting.**

## ⏱ Оптимизация памяти и ускорение вычислений

### Оптимизация загрузки
#### Работа с `pickle`
`Pickle` - это отличная альтернатива привычным нам `.csv` файлам при работе с большими файлами. Мало того, что он считывает и сохраняет всё в разы быстрее, так еще и место на диске такой файл занимает меньше. Также, при использовании `to_pickle()`, сохраняются индексы и все типы колонок, так что при его последующем считывании, датафрейм будет точно таким же и его не нужно будет повторно оптимизировать при каждом открытии, как при использовании `CSV` формата

```
pd.to_pickle()
pd.read_pickle()
```
#### Считывание по батчам
```
import gc

chunksize = 1000
tmp_lst = []
with pd.read_csv('../data/car_train.csv',
                 index_col='car_id',
                 dtype={'model': 'category',
                        'car_type': 'category',
                        'fuel_type': 'category',
                        'target_class': 'category'}, chunksize=chunksize) as reader:
    for chunk in reader:
        tmp_lst.append(chunk)
        
data = pd.concat(tmp_lst)

del tmp_lst
gc.collect()

data.head()
```

**Важно:** Пока на какой-то объект в памяти есть ссылки, которые явно не удалили или не переназначили - этот объект будет занимать оперативную память, хотя он может больше не использоваться. Поэтому все временные объекты, которые больше не будут использоваться, лучше явно удалять, используя `del` и после этого запускать сборщик мусора `gc` - `garbage collector`, как в примере выше.

#### Using a generator
```
def read_file(filename):
    with open(filename, 'r') as f:
        for line in f:
            yield line.strip()
```
```
it = read_file('../data/car_info.csv')
next(it)
```




### 🗜 Оптимизация памяти
#### Оптимизация числовых типов
```
def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype.name

        if col_type not in ['object', 'category', 'datetime64[ns, UTC]']:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
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

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df
```
And then: 
```
df = reduce_mem_usage(df)
df.info()
```
#### Оптимизация категориальных фичей
```
def convert_columns_to_catg(df, column_list):
    for col in column_list:
        print("converting", col.ljust(30), "size: ", round(df[col].memory_usage(deep=True)*1e-6,2), end="\t")
        df[col] = df[col].astype("category")
        print("->\t", round(df[col].memory_usage(deep=True)*1e-6,2))
        
```
###  🥌 Ускорение при помощи `Numpy`
```
a = list(range(1_000_000))
b = np.arange(1_000_000)
```
Примеры:
```
10000 in b < 10000 in a

b + 10 < [el+10 for el in a]

b.max() < max(b)
```
Если нужно проверить наличие элементов в другом массиве и тот "другой" достаточно большой, то лучше использовать `set()`, так как в нем поиск элемента осуществляется за `O(log(n))`, а в `np.isin()` за `O(n)`. Так что, даже несмотря на то, что `numpy` оптимизирован, `set()` выигрывает его по времени.
```
st=set(b)
[el in st for el in a] < np.isin(a,b)

np.where() < if-function

np.vectorize() > np.where()

np.select(conditions, choices, default=3) -> like np.where weith multiple conditions

np.where(df_cars['car_rating'] > 5, df_cars['car_type'].map(mydict), np.nan) < apply mapping function to df
```
**Ускоряем groupby**
1. Считаем кол-во значений в группe
```
df_cars.groupby(['model', 'fuel_type'])['target_reg'].count()



df_cars['int_model'] = lbl.fit_transform((df_cars['model'] + df_cars['fuel_type']).astype(str))
np.bincount(df_cars['int_model'])
```
2. Сумма/среднее
```
df_cars.groupby(['model', 'fuel_type'])['target_reg'].sum()

np.bincount(df_cars['int_model'], weights=df_cars['target_reg'])
```
3. Min/max
```
df_cars.groupby(['int_model'])['target_reg'].agg(['max'])

indices = df_cars['int_model']
max_values = np.maximum.reduceat(df_cars['target_reg'].values[np.argsort(indices)],
                                 np.concatenate(([0], np.cumsum(np.bincount(indices))))[:-1])
```

### ⚡️ `Numba Jit`
Simple example: 
```
from numba import jit

@jit(nopython=True)
def f(n):
    s = 0.
    for i in range(n):
        s += i ** 0.5
    return s
```
Example with pandas dataframe:
```
@jit(nopython=True)
def monotonically_increasing(a):
    max_value = 0
    for i in range(len(a)):
        if a[i] > max_value:
            max_value = a[i]
        a[i] = max_value
    return a
    
%%timeit
monotonically_increasing(df_cars['target_reg'].values)
```
### 🧵 Multiprocessing
С помощью `multiprocessing`, можно ускорить вообще практически все. Он позвлоляет использовать не одно ядро вашего компьютера, а сразу несколько и, соответственно, ускорить вычисления (уже не только `io-bound`, но еще и `cpu-bound`) в количество раз, пропорциональное количеству доступных ядер.

```
def parallelize_dataframe(df, func, n_cores=4):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = np.concatenate(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df
%%time
tmp = parallelize_dataframe(df, text_prepare.many_row_prepare)
```

##  🚣 Парсинг внешних данных
### Анализ URL запроса
Рассмотрим для начала структуру URL адреса, это важно! URL адрес имеет определенную структуру, которая включает:

- метод доступа к ресурсу, который также называется сетевым **протоколом**;
- **авторизацию доступа**;
- **хосты** – DNS адрес, который указан как IP адрес;
- **порт** – еще одна обязательная деталь, которая включается в сочетание с IP адресом;
- **трек** – определяет информацию о методе получения доступа;
- **параметр** – внутренние данные ресурса о файле.

```
from urllib.parse import urlparse, parse_qsl, parse_qs
url = "http://www.example.com:80/path/to/myfile.html?key1=value1&key2=value2#SomewhereInTheDocument"
url_parsed = urlparse(url)
url_parsed
```
Parse result: `ParseResult(scheme='http', netloc='[www.example.com:80](http://www.example.com/)', path='/path/to/myfile.html', params='', query='key1=value1&key2=value2', fragment='SomewhereInTheDocument')`

The example of POST request:
```
page = requests.post('https://controlc.com/index.php?act=submit', data={
    'subdomain': '',
    'antispam': 1,
    'website': '',
    'paste_title': 'Заметка',
    'input_text': 'Привет!',
    'timestamp': 'ba68753935524ba7096650590c86633b',
    'paste_password': '',
    'code': 0,
}, headers={'accept-encoding': 'identity', 'referer': 'https://controlc.com/'})
page
```
### Parsing of HTML page with BeautifulSoup
```
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
from operator import attrgetter, itemgetter

soup = BeautifulSoup(open('../data/parsing_data/689066_2.html', 'rb').read(), 'lxml')
```
Берем основную информацию со странички. Запустив сессию `BeautifulSoup`, найдем теги, здесь это можно гораздо проще и приятнее для глаз - `.find("tag", attribute="value")`. Для того чтобы найти все встречающиеся варианты используем `.find_all("tag")`
```
desc = soup.find('div', itemprop='description')
desc = soup.find('div', class_=lambda s: s and s.startswith("styles_synopsisSection")).find_all('p')

film_info = {
    'title': soup.find('h1', itemprop='name').find('span').text,
    'title-original': soup.find('span', class_=lambda s: s and s.startswith('styles_originalTitle__')).text,
    'rating': float(soup.find('a', {'class': 'film-rating-value'}).text),
    'description': '\n'.join(map(attrgetter('text'), desc))
}
film_info
```
### Selenium



## 🐍🔥 Двухголовая нейронка на `PyTorch`.
### Self-implemented

```
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn
import copy

train, test = train_test_split(df, test_size=0.2, random_state=42)

device = torch.device('cpu')

# ВАЖНО! - фиксируем воспроизводимость
def seed_everything(seed=42):
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(seed=42)
```
Make a configuration class:
```
class CFG:
    hidden_size=128
    dropout=0.1
    lr=1e-3
    batch_size=128
    num_workers=4
    epochs=20
    num_features=train.shape[1]-2
    num_tar_class=train.target_class.nunique() 
```
A dataset for our table:
```
class Rides(Dataset):
    def __init__(self, df):
        self.df = df
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx,:]
        data = row.drop(labels=['target_reg', 'target_class'])
        data = torch.FloatTensor(data.values.astype('float'))
        tar_reg = torch.tensor(row['target_reg']).float()
        tar_class = row['target_class'].astype('int')
        
        return data, tar_reg, tar_class
train_datasets = {'train': Rides(train),
                  'val': Rides(test)}
```
Dataloaders:
```
dataloaders_dict = {x: torch.utils.data.DataLoader(train_datasets[x], batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers)
	for x in ['train', 'val']}
```
Now the network:
```
class TabularNN(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mlp = nn.Sequential(
                          nn.Linear(cfg.num_features, cfg.hidden_size),
                          #nn.BatchNorm1d(cfg.hidden_size),
                          nn.Dropout(cfg.dropout),
                          nn.ReLU(),
                          nn.Linear(cfg.hidden_size, cfg.hidden_size),
                          #nn.BatchNorm1d(cfg.hidden_size),
                          nn.Dropout(cfg.dropout),
                          nn.ReLU(),
                          nn.Linear(cfg.hidden_size, cfg.hidden_size//2),
                          )
        
        self.regressor = nn.Sequential(
            nn.Linear(cfg.hidden_size // 2, 1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(cfg.hidden_size // 2, cfg.num_tar_class)
        )

    def forward(self, data):
        x = self.mlp(data)
        tar_reg = self.regressor(x)
        tar_class = self.classifier(x)
        return tar_reg.view(-1), tar_class
```
Initialize it and optimizer/critetion:
```
model = TabularNN(CFG).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr = CFG.lr)
regression_criterion = nn.MSELoss().to(device)
classification_criterion = nn.CrossEntropyLoss().to(device)
```
Training code:
```
def train_model(model, dataloaders, regression_criterion,
                classification_criterion, optimizer, num_epochs=25,
                early_stopping_rounds=5, verbose=2):

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.inf
    early_steps = 0
    stop = False

    for epoch in range(num_epochs):
        if stop:
            break
        if epoch % verbose == 0:
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            for inputs, labels_1, labels_2 in dataloaders[phase]:
                inputs = inputs.to(device)
                labels_1 = labels_1.to(device)
                labels_2 = labels_2.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss

                    outputs_1, outputs_2 = model(inputs)
                    loss_1 = regression_criterion(outputs_1, labels_1)
                    loss_2 = classification_criterion(outputs_2, labels_2)

                    loss = loss_1 + loss_2

                    _, preds_2 = torch.max(outputs_2, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        #scheduler.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                val_acc_history.append(running_loss)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            if epoch % verbose == 0:
                print('{} Loss: {:.4f}'.format(phase, epoch_loss))

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_model_wts = copy.deepcopy(model.state_dict())
                best_loss = epoch_loss
                early_steps = 0
            if phase == 'val' and epoch_loss > best_loss:
                early_steps += 1
                if early_steps > early_stopping_rounds:
                    stop = True
                    print(f'Stopped by early_stopping. Epoch: {epoch}')
                    break 
    # load best model weights
    model.load_state_dict(best_model_wts)
    return model
    
model_ft = train_model(model, dataloaders_dict, regression_criterion,
                classification_criterion, optimizer, num_epochs=22)
```
Onto predictions:
```
# p1, p2 - предсказания; l1, l2  - истинные значения

p1, p2, l1, l2 = [], [], [], []

with torch.set_grad_enabled(False):
    # Get model outputs and calculate loss
    for inputs, labels_1, labels_2 in dataloaders_dict['val']:
        inputs = inputs.to(device)
        labels_1 = labels_1.to(device)
        labels_2 = labels_2.to(device)
        l1.extend(labels_1.detach().cpu().numpy())
        l2.extend(labels_2.detach().cpu().numpy())
        
        outputs_1, outputs_2 = model_ft(inputs)
        _, outputs_2 = torch.max(outputs_2, 1)

        p1.extend(outputs_1.detach().cpu().numpy())
        p2.extend(outputs_2.detach().cpu().numpy())
        
torch.save(model_ft.state_dict(), 'tab_model.pth')
```
##   🕸 `TabNet` - "SOTA" для табличек.
```
y = df['target_class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from pytorch_tabnet.tab_model import TabNetClassifier

clf = TabNetClassifier(device_name='cpu')
clf.fit(X_train.values,
        y_train,
        patience=100,
        eval_set=[(X_test.values, y_test)])
```
We can also view feature importance:
```
for i, j in sorted(zip(clf.feature_importances_.astype('float16'), X_train.columns), reverse=True):
    print(i, j)
```
[Пример интерпретации TabNet](https://www.kaggle.com/code/carlmcbrideellis/tabnet-and-interpretability-jane-street-example)
```
import matplotlib.pyplot as plt

explain_matrix, masks = clf.explain(X_test.values)

fig, axs = plt.subplots(1, 3, figsize=(20,20))
for i in range(3):
    axs[i].imshow(masks[i][:50])
    axs[i].set_title(f"mask {i}")
    
```

```
normalized_explain_mat = np.divide(explain_matrix, explain_matrix.sum(axis=1).reshape(-1, 1)+1e-8)

# Add prediction to better understand correlation between features and predictions
preds = clf.predict(X_test.values)

explain_and_preds = np.hstack([normalized_explain_mat, preds.reshape(-1, 1)])

correlation_importance = np.corrcoef(explain_and_preds.T)
px.imshow(correlation_importance,
          labels=dict(x="Features", y="Features", color="Correlation"),
          x=list(X_test.columns)+["prediction"], y=list(X_test.columns)+["prediction"],
          title="Correlation between attention mechanism for each feature and predictions",
          width=1000,
          height=1000,
          color_continuous_scale='Jet')
```


## 🚚 Extracting embeddings
### TabularNetworks
Just return the result from last layer without classification/regression head
```
# Загружаем, сохранённые ранее, веса модели
model = TabularNN(CFG)
PATH = 'tab_model.pth'
model.load_state_dict(torch.load(PATH))

# Обычно, для получения эмбеддингов, финальные слои заменяют на nn.Identity()
# В нашем случае, это делать необязательно
model.classifier = torch.nn.Identity()
model.regressor = torch.nn.Identity()
model.to(device)
model.eval()
```
Now extract embeddings:
```
embeddings = []

with torch.no_grad():
    for inputs, labels_1, labels_2 in dataloader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        embeddings.extend(outputs.detach().cpu().numpy())
embed_df = pd.DataFrame(data=embeddings,columns=[f'embed_{i}' for i in range(embeddings[0].shape[0])])

df = pd.concat((df, embed_df) ,axis=1)
df.head()
```
Train CatBoost on that data
### LLMs

####  🧵 TextEmbeddings
```
class TextEmbeddings:
    def __init__(self, add_cls_embeddings=True, add_mean_embeddings=False):
        self.add_mean_embeddings = add_mean_embeddings
        self.add_cls_embeddings = add_cls_embeddings
        if add_cls_embeddings is False and add_mean_embeddings is False:
            raise 'Error: you should select at least one type of embeddings to be computed'

    def mean_pooling(self, hidden_state, attention_mask):
        """
        Возвращает усредненный с учетом attention_mask hidden_state.
        """
        token_embeddings = hidden_state.detach().cpu() 
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        return sum_embeddings / attention_mask.sum()

    def extract_embeddings(self, texts, model_name, max_len):
        """
        Возвращает значения, посчитанные данной моделью - эмбеддинги для всех текстов из texts.
        """
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name).cuda()
        text_features = []
        for sentence in tqdm(texts):
            encoded_input = tokenizer([sentence],
                                      padding='max_length',
                                      truncation=True,
                                      max_length=max_len,
                                      return_tensors='pt')
            with torch.no_grad():
                hidden_state, cls_head = model(input_ids=encoded_input['input_ids'].cuda(), return_dict=False)
                sentence_embeddings = self.mean_pooling(hidden_state, encoded_input['attention_mask'])
            
            now_emb = []
            if self.add_cls_embeddings:
                now_emb.append(cls_head.detach().cpu().numpy().flatten())
            
            if self.add_mean_embeddings:
	            now_emb.append(sentence_embeddings.detach().cpu().numpy().flatten())
            
            text_features.append(np.concatenate(now_emb, axis=0))
        return text_features

    def add_many_embeddings(self, df, text_col, models):
        """"
        Добавляет в качестве признаков эмбеддинги для колонки text_col.
        В качестве моделей и максимальных длин используются models.
        """
        for model_name, max_len in models:
            print(model_name)
            text_features = self.extract_embeddings(df[text_col], model_name, max_len)
            text_features_df = pd.DataFrame(text_features, columns = [f'{model_name}_{text_col}_feature_{i}' for i in range(len(text_features[0]))])
            df = df.join(text_features_df)
            df.to_csv('transformers_text_features.csv', index=False)
            os.system('cp /content/transformers_text_features.csv /content/drive/MyDrive/datasets/transformers_text_features.csv')
        return df
```

#### 🗣 Советы по использованию языковых моделей

Напоследок, несколько советов по работе с текстами.

**Когда применять языковые модели:**

1. Тексты не очень большие (в идеале до 512 токенов)
2. Вы хотите опираться на смысл текста, а не на ключевые слова
3. Есть GPU ресурсы и от вас не требуют мгновенной скорости работы

**Как выбрать какие модели использовать:**

1. Лучше всего использовать модели, предобученные на вашем домене, если такие есть
2. Обратите внимание, что ваша модель работает с необходимым вам языком
3. Если очень хочется, то тексты можно перевести на английский (см. урок про парсинг) и использовать модели уже для переведнных текстов
4. Модели можно дообучать на вашей задаче
5. Можно доставать эмбеддинги и использовать их в качестве признаков, а можно просто предсказывать таргет с помощью языковой модели
6. Если есть время и мощности, то не пренебрегайте стекингом разных языковых моделей


## 🏋️‍♂️🏌️‍♂️ Weigths & Biases
#### 🔑 Ключевые особенности

- Позволяет отслеживать показатели работы модели в режиме реального времени и сразу же выявлять проблемные места.
- Визуализация результатов - графики, изображения, видео, аудио, 3D-объекты и многое [другое](https://docs.wandb.ai/guides/track/log#logging-objects).
- Позволяет тренировать модели с разных устройств и хранить результаты в одном месте, также удобно при командной работе над задачей
- Использование артефактов `W&B` позволяет отслеживать и создавать версии ваших наборов данных, моделей, зависимостей и результатов в конвейерах машинного обучения.
- Низкий порог входа (можно начать с 6 строк кода), также есть готовые интеграции со всеми популярными DS фреймворками

#### 📲 Регистрация учетной записи и вход в систему
На `Kaggle` можно авторизоваться в `W&B` двумя способами:

1. С помощью `wandb.login ()`. Он запросит ключ API, который вы можете скопировать + вставить.
2. Используя `Kaggle secrets` для хранения ключа API и использовать приведенный ниже фрагмент кода для входа в систему. 
```
from kaggle_secrets import UserSecretsClient

user_secrets = UserSecretsClient()
wandb_api = user_secrets.get_secret("wandb_api") 
wandb.login(key=wandb_api)
```
####  ✍️ Залогируем, для примера, реализацию `MLP`
```
import torch
import os
import copy
import sys
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
```
Save hyperparameters for later use: 
```
CONFIG = dict (
    hidden_size=128,
    dropout=0.1,
    lr=1e-3,
    batch_size=128,
    num_workers=os.cpu_count(),
    epochs=20,
    num_features=train.shape[1]-2, # количество фичей, подаваемое на вход
    num_tar_2=train.target_class.nunique(), # количество выходов равно количеству предсказываемых классов
    architecture = "MLP",
    infra = "Kaggle"
)
# Инициализируем модель
model = TabularNN(CONFIG).to(device)

# оптимайзер и лоссы для регрессии и классификации
optimizer = torch.optim.Adam(model.parameters(), lr = CONFIG['lr'])
regression_criterion = nn.MSELoss().to(device)
classification_criterion = nn.CrossEntropyLoss().to(device)
```
Training 5 epochs with windb:
```
import random

num_epochs = CONFIG['epochs']
verbose = 5

for _ in range(5):
    
    # Изменим CONFIG - определим dropout как рандомную величину в заданном диапазоне значений
    CONFIG['dropout'] = random.uniform(0.01, 0.80)
    model = TabularNN(CONFIG).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = CONFIG['lr'])
    
    # 🐝 initialize a wandb run
    wandb.init(project='Course contest',
               config=CONFIG,
               group='MLP', 
               job_type='train'
            )
    
    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = np.inf

    for epoch in range(num_epochs):
        if epoch % verbose == 0:
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            for inputs, labels_1, labels_2 in dataloaders[phase]:
                inputs = inputs.to(device)
                labels_1 = labels_1.to(device)
                labels_2 = labels_2.to(device)
                
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss

                    outputs_1, outputs_2 = model(inputs)
                    loss_1 = regression_criterion(outputs_1, labels_1)
                    loss_2 = classification_criterion(outputs_2, labels_2)

                    loss = loss_1 + loss_2

                    _, preds_2 = torch.max(outputs_2, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                val_acc_history.append(running_loss)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            if epoch % verbose == 0:
                print('{} Loss: {:.4f}'.format(phase, epoch_loss))
            
            # 🐝 Log train and validation metrics to wandb
            wandb.log({'{} loss'.format(phase): epoch_loss, 'dropout':CONFIG['dropout']})

            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                best_model_wts = copy.deepcopy(model.state_dict())
                best_loss = epoch_loss

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    # 🐝 Close your wandb run 
    wandb.finish()
```
#### 💎 Создание артефакта `W&B`.
`W&B Artifacts` позволяет вам регистрировать данные, которые входят (например, набор данных) и выходят (например, веса обученной модели) из этих процессов.

Другими словами, артефакты - это способ сохранить ваши наборы данных и модели. Вы можете использовать [этот Colab](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/wandb-artifacts/Pipeline_Versioning_with_W%26B_Artifacts.ipynb), чтобы узнать больше об артефактах.

##### Сохраняем свою работу с помощью `wandb.log_artifact ()`

В `run` есть три шага для создания и сохранения артефакта модели.

1. Создайте пустой артефакт с помощью `wandb.Artifact ()`.
2. Добавьте файл модели в Артефакт с помощью `wandb.add_file ()`.
3. Вызовите `wandb.log_artifact ()`, чтобы сохранить Артефакт.

##### Пример
```
# Save model
torch.save(model.state_dict(), 'tab_model.pth')

# Initialize a new W&B run
run = wandb.init(project='Course contest',
                 config=CONFIG,
                 group='MLP', 
                 job_type='save') # Note the job_type

# Update `wandb.config`
wandb.config.type = 'baseline'
wandb.config.kaggle_competition = 'Competitive Data Science Course'

# Save model as Model Artifact
artifact = wandb.Artifact(name='best_tab_NN', type='model') # Задаем произвольное имя
artifact.add_file('tab_model.pth')
run.log_artifact(artifact)

# Finish W&B run
run.finish()
```
## 🎲 Работа с метрикой. Пре-процессинг и пост-процессинг

1. Чекаем распределение таргета
2. Чекаем распределение классов в предикте
3. Дефолтные пороги - порок
4. Не округляй, если AUC
	Мы часто замечали проблему округления у новичков, которая сильно может подкосить в начале. Допустим в вашем чемпионате метрика `roc-auc`.  Если у вас тестирующая метрика на борде `roc-auc`, то при отправке ответов не надо переводить вероятности в классы. Это всегда дает результат хуже, чем сами вероятности. Поэтому проверяйте, что вы засылаете вероятности!
5. Обратная вероятность или `1 - P`

##  📷 Multi-threaded video processing
```
# importing required libraries 
import cv2 
import time 
from threading import Thread # library for implementing multi-threaded processing 

# defining a helper class for implementing multi-threaded processing 
class WebcamStream :
    def __init__(self, stream_id=0):
        self.stream_id = stream_id   # default is 0 for primary camera 
        
        # opening video capture stream 
        self.vcap      = cv2.VideoCapture(self.stream_id)
        if self.vcap.isOpened() is False :
            print("[Exiting]: Error accessing webcam stream.")
            exit(0)
        fps_input_stream = int(self.vcap.get(5))
        print("FPS of webcam hardware/input stream: {}".format(fps_input_stream))
            
        # reading a single frame from vcap stream for initializing 
        self.grabbed , self.frame = self.vcap.read()
        if self.grabbed is False :
            print('[Exiting] No more frames to read')
            exit(0)

        # self.stopped is set to False when frames are being read from self.vcap stream 
        self.stopped = True 

        # reference to the thread for reading next available frame from input stream 
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True # daemon threads keep running in the background while the program is executing 
        
    # method for starting the thread for grabbing next available frame in input stream 
    def start(self):
        self.stopped = False
        self.t.start() 

    # method for reading next frame 
    def update(self):
        while True :
            if self.stopped is True :
                break
            self.grabbed , self.frame = self.vcap.read()
            if self.grabbed is False :
                print('[Exiting] No more frames to read')
                self.stopped = True
                break 
        self.vcap.release()

    # method for returning latest read frame 
    def read(self):
        return self.frame

    # method called to stop reading frames 
    def stop(self):
        self.stopped = True 


# initializing and starting multi-threaded webcam capture input stream 
webcam_stream = WebcamStream(stream_id=0) #  stream_id = 0 is for primary camera 
webcam_stream.start()

# processing frames in input stream
num_frames_processed = 0 
start = time.time()
while True :
    if webcam_stream.stopped is True :
        break
    else :
        frame = webcam_stream.read() 

    # adding a delay for simulating time taken for processing a frame 
    delay = 0.03 # delay value in seconds. so, delay=1 is equivalent to 1 second 
    time.sleep(delay) 
    num_frames_processed += 1 

    cv2.imshow('frame' , frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
end = time.time()
webcam_stream.stop() # stop the webcam stream 

# printing time elapsed and fps 
elapsed = end-start
fps = num_frames_processed/elapsed 
print("FPS: {} , Elapsed Time: {} , Frames Processed: {}".format(fps, elapsed, num_frames_processed))

# closing all windows 
cv2.destroyAllWindows()
```

## ❓❓Things to consider!
- DuckDB : через  DuckDB очень изящно выдергивается нужное из неподъемного датасета, который в память не лезет. Не видел такое раньше, потому расшарил тут
- RAPIDS

## 📷 Torchvision video processing

[Torchvision](https://github.com/pytorch/vision) implemented a video reader that helps decode videos using the GPU. It is still in Beta stage, and requires some work to set up, as you must build it from source. The steps are pretty straightforward: pre-install FFmpeg, download [Nvidia’s Video Codec SDK](https://developer.nvidia.com/nvidia-video-codec-sdk), and make sure you’re building the package with a matching Torch version. The end result is a Torchvision package with GPU video decoding capabilities that’s super easy to use. You can then take this package and publish it to your local repository, making it accessible to everyone on your team.

```
import torch
import torchvision
 
 
def preprocess_video_gpu(video_path, image_size_low, frames_per_cycle=30):
   reader = torchvision.io.VideoReader(video_path, "video", num_threads=0, device="cuda")
   resizer = torchvision.transforms.Resize(image_size_low, antialias=True)
 
   curr_frames = []
   all_frames_low = []
 
   for frame in reader:
       curr_frames.append(frame["data"])
 
       if len(curr_frames) == frames_per_cycle:
           resize_chunk(curr_frames, all_frames_low, resizer)
           curr_frames = []
 
   if len(curr_frames) > 0:
       resize_chunk(curr_frames, all_frames_low, resizer)
 
   all_frames_low = torch.cat(all_frames_low, 0)
 
   return all_frames_low
 
def resize_chunk(curr_frames, all_frames_low, resizer):
   curr_frames = torch.stack(curr_frames, 0)
   curr_frames = curr_frames.permute(0, 3, 1, 2)
   curr_frames_low = resizer(curr_frames)
   curr_frames_low = curr_frames_low.permute(0, 2, 3, 1)
   all_frames_low.append(curr_frames_low)
 
print("Preprocessing on GPU...")
s = time.time()
frames = preprocess_video_gpu(video_path=VIDEO_PATH, image_size_low=IMAGE_SIZE_LOW)
print("Preprocess on GPU time:", time.time() - s)
```
This code initializes the VideoReader object, then reads frames into a list. Keep in mind, we want to perform the resize on big batches of frames, for better performance. However, holding all the full-size frames in GPU memory might cause us to crash, depending on the video length and the amount of GPU memory available, so we use the frames_per_cycle parameter. Every frames_per_cycle frames, we resize the current frames, and move on to the next batch. Depending on the GPU used, and the video’s original resolution, we can fine-tune this parameter. I used a T4 GPU



# Making Deep Learning Go Brrrr From First Principles

Just like with training ML models, knowing what regime you're in allows you to narrow in on optimizations that matters. For example, if you're spending all of your time doing memory transfers (i.e. you are in an _memory-bandwidth bound_ regime), then increasing the FLOPS of your GPU won't help. On the other hand, if you're spending all of your time performing big chonky matmuls (i.e. a _compute-bound_ regime), then rewriting your model logic into C++ to reduce overhead won't help.

So, if you want to keep your GPUs going brrrr, let's discuss the three components your system might be spending time on - compute, memory bandwidth, and overhead.

## Compute

One perspective on optimizing deep learning systems is that we'd like to maximize the time in the compute-bound regime. You paid for all of those 312 teraflops, and ideally, you'd _get_ those 312 teraflops. But, in order to get your money's worth out of your expensive matrix multiplication, you need to reduce the amount of time spent in the other parts.

But why the focus on maximizing compute and not say, memory bandwidth? The reason is simple - you can reduce the overhead or memory costs, but you (mostly) can't reduce the computation required without changing the actual operations you're performing.

Exacerbating the difficulty of maximizing compute utilization is the rate at which compute grows compared to memory bandwidth
![[Pasted image 20240614113005.png]]


One way to think about compute is as a factory. We send instructions to our factory (overhead), send it materials (memory-bandwidth), all to keep our factory running efficiently (compute).
So, if our factory increases efficiency faster than the rate at which we can supply it materials, it becomes harder for our factory to achieve its peak efficiency.

 If you aren't doing matrix multiplication, you'll only be able to achieve 19.5 teraflops instead of the stated 312. Note that this isn't unique to GPUs - in fact, TPUs are even _less_ general than GPUs.
The fact that GPUs are so much slower at everything that isn't a matrix multiply might seem problematic at first - what about our other operators like layer norm or activation functions? Well, the truth is, those operators are just rounding errors in terms of FLOPS. For example, let's look at this table of FLOP counts on BERT for different operator types from [this paper](https://arxiv.org/abs/2007.00072), where "Tensor Contraction" = matmuls.

![[Pasted image 20240614113646.png]]You can see that altogether, our non-matmul ops only make up 0.2% of our FLOPS, so it doesn't matter that our GPU computes non-matmul ops 15x slower.
But, in this case, the normalization and pointwise ops actually achieve **250x less FLOPS and 700x less FLOPS** than our matmuls respectively.

So why do our non-matmul ops take so much more time than they should?

Going back to our analogy, the culprit is often how long it takes to transport materials to and from the factory. In other words, the memory bandwidth.

## Bandwidth

Bandwidth costs are essentially the cost paid to move data from one place to another. This might be moving the data from CPU to GPU, from one node to another, or even from CUDA global memory to CUDA shared memory. This last one, in particular, is what we'll be focusing on here, and is **typically referred to as "bandwidth cost" or "memory bandwidth cost".**

Although our factory is where we do the actual work, it's not suitable as a bulk storage unit. A large part of this is that since we're doing actual work here, all the storage is optimized for being fast to actually _use_ (SRAM), instead of having a lot of it.

So, where do we store the actual results and materials? The typical approach is to have a warehouse, probably somewhere where land is cheap and we have a lot of space (DRAM). Then, we can ship supplies to and from our factories (memory bandwidth).

As an aside, your GPU's DRAM is what shows up in `nvidia-smi`, and is the primary quantity responsible for your lovely "CUDA Out of Memory' errors.

Now, imagine what happens when we perform an unary operation like `torch.cos`. We need to ship our data from our storage to the warehouse, then perform a tiny bit of computation for each piece of data, and then ship that storage back. Shipping things around is quite expensive. As a result, nearly all of our time here is spent shipping data around, and _not_ on the actual computation itself.

Since we're spending all of our time on memory-bandwidth, such an operation is called a **memory-bound operation**, and it means that we're not spending a lot of time on compute.

operator fusion - the most important optimization in deep learning compilers. Simply put, instead of writing our data to global memory just to read it again, we elide the extra memory accesses by performing several computations at once.

For example, if we perform `x.cos().cos()`, usually we need to perform 4 global reads and writes.

```
x1 = x.cos() # Read from x in global memory, write to x1
x2 = x1.cos() # Read from x1 in global memory, write to x2
```
But, with operator fusion, we only need 2 global memory reads and writes! So operator fusion will speed it up by 2x.
```
x2 = x.cos().cos() # Read from x in global memory, write to x2
```

Not all operator fusion is as simple as pointwise operators. You can fuse pointwise operators onto reductions, or pointwise operators onto matrix multiplication. Even matrix multiplication itself can be thought of as fusing a broadcasting multiply followed by a reduction.

Finally, operator fusion leads to some surprising consequences. For one, a fused `x.cos().cos()` will take nearly the exact same time as calling `x.cos()` by itself. This is why activation functions are nearly all the same cost, despite `gelu` obviously consisting of many more operations than `relu`.

This fact leads to some interesting consequences for rematerialization/activation checkpointing. Essentially, doing extra recomputation might lead to _less_ memory-bandwidth, and thus less runtime. Thus, we can lower both memory _and_ runtime through rematerialization, which we leveraged to build a neat min-cut optimization pass in AOTAutograd. You can read more about it [here](https://dev-discuss.pytorch.org/t/min-cut-optimal-recomputation-i-e-activation-checkpointing-with-aotautograd/467/1)

## Reasoning about Memory-Bandwidth costs

When it come to reasoning about whether your operation is memory-bandwidth bound, a calculator can go a long way.

For simple operators, it's feasible to reason about your memory bandwidth directly. For example, an A100 has 1.5 terabytes/second of global memory bandwidth, and can perform 19.5 teraflops/second of compute. So, if you're using 32 bit floats (i.e. 4 bytes), you can load in 400 billion numbers in the same time that the GPU can perform 20 trillion operations. Moreover, to perform a simple unary operator (like multiplying a tensor by 2), we actually need to _write_ the tensor back to global memory.

So... until you're doing about a hundred operations in your unary operator, you'll be spending more time performing memory accesses than actual compute.

With the help of a fusing compiler like NVFuser, it's actually fairly easy to measure this ourselves! You can see the code in Colab [here](https://colab.research.google.com/drive/1hEtorT5y9mcXHR0gpensD7oZfuyyxtu7?usp=sharing).

If you take a PyTorch function like
```
def f(x: Tensor[N]):
    for _ in range(repeat):
        x = x * 2
    return x
```

and benchmark it with a fusing compiler, we can then calculate the FLOPS and memory bandwidth achieved for various values of `repeat`. Increasing `repeat` is an easy way of increasing our amount of compute _without_ increasing our memory accesses - this is also known as increasing **compute intensity**.

## Overhead
Overhead is when your code is spending time doing anything that's **not** transferring tensors or computing things. For example, time spent in the Python interpreter? Overhead. Time spent in the PyTorch framework? Overhead. Time spent launching CUDA kernels (but not executing them)? Also overhead.

The primary reason overhead is such a pernicious problem is that modern GPUs are _really_ fast. An A100 can perform 312 **trillion** floating point operations per second (312 TeraFLOPS). In comparison, Python is _really_ slooooowwww. Benchmarking locally, Python can perform 32 million additions in one second.

That means that in the time that Python can perform a _single_ FLOP, an A100 could have chewed through **9.75 million FLOPS**.

Even worse, the Python interpreter isn't even the only source of overhead - frameworks like PyTorch also have many layers of dispatch before you get to your actual kernel. If you perform the same experiment with PyTorch, we can only get 280 thousand operations per second. Of course, tiny tensors aren't what PyTorch is built for, but... if you are using tiny tensors (such as in scientific computing), you might find PyTorch incredibly slow compared to C++.

Given this, you might be shocked that anybody uses PyTorch at all, but keep in mind that modern deep learning models are often performing **massive** operations. Moreover, frameworks like PyTorch execute _asynchronously_. That is, while PyTorch is running a CUDA kernel, it can continue and queue up more CUDA kernels behind it. So, as long as PyTorch can "run ahead" of the CUDA kernels, most of the framework overhead gets completely hidden!

**So, how do you tell if you're in this regime? Well, since overhead generally doesn't scale with problem size (while compute and memory do), the easiest way to tell is to simply increase the size of your data. If that doesn't increase the runtime proportionally, you're overhead bound. For example, if you double your batch size but your runtime only increases by 10%, you're likely overhead bound.**

Another aside - the "GPU-Util" ([not "Volatile GPU-Util"](https://twitter.com/cHHillee/status/1500547396945670144)) entry in nvidia-smi is basically measuring what percentage of the bottom row is actually running a GPU kernel. So that's another good way of eyeballing overhead.

Fundamentally, this overhead comes from the flexibility of being able to do something different at each step. If you don't need this flexibility, one way of resolving this flexibility is by tracing it out, like with `jit.trace`, `FX`, or `jax.jit`. Or, alternately, you could do it at an even lower level with something like [CUDA Graphs](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/).

## Conclusion
