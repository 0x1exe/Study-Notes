### RMSLE metric 

Defined as: 
$$
\text{RMSLE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (\log(p_i + 1) - \log(a_i + 1))^2}
$$
Constant predictions:
- Constant predictions are a goocd way to get a sense of what it means to have a good performance on model
- Constant score for RMSLE is: ```e**(mean(ln(targets))) ``` this is an optimal score for naive baseline, a model should perform at least better than this.
- RMSLE only considers the relative error between [p] and [a] and the scale of error is insignificant (While RMSE value increases in magnitude with scale increase)
- RMSLE incurs a larger penalty for underestimation

### Quadratic weighted kappa
- Ranges between [-1,1]
- Computational steps: ```1. Confusion matrix 2.difference_matrix 3.value_counts() for preds and actual 4. E = outer_prod of value vectors 5.Normalize E and O matrices 6. Compute using formula:```
$$
\text{Quadratic Weighted Kappa} = 1 - \frac{\sum_{i,j} w_{ij} O_{ij}}{\sum_{i,j} w_{ij} E_{ij}}
$$

### ROC-AUC
- ROC stands for 'Receive operating characteristics'
- AUC stands for 'Area under the curve'
- Used for classification

- ROC -> probability curve
- AUC -> measure of separability
- Higher AUC -> better class separation
- ROC is obtained by plotting TPR vs. FPR
TPR or recall is computed by:
```
TP/(TP + FN)
```
And FPR is:
```
Specificity = TN/(TN+FP)
FPR = 1 - specificity = FP/(TN+FP)
```
- AUC is scale-invariant and classification-threshold-invariant
$$
\text{ROC-AUC} = \int_{0}^{1} \text{TPR}(\text{FPR}^{-1}(t)) \, dt
$$
### Log-cosh loss
- Properties stem from hyperbolic cosine

$$
\text{Log-Cosh Loss} = \frac1n \sum_{i=1}^{n} \log(\cosh(p_i - a_i))
$$
#### Advantages:
- Smoothness
- Robust
- Near-quadratic behaviour
#### Disadvantages: 
- Complexity
- Less comonly used

#### Differentiation

$$
1. L(y-p) = \sum \log(\cosh(y-p)) )
$$
Turn sum into a power:
$$
2. ( \log(\cosh^n(y-p)) )
$$
$$
3. ( L'(y-p) = n \cdot \tanh(y-p) )
$$
$$
4. ( L''(y-p) = n \cdot \frac{1}{{\cosh^2(y-p)}} )
$$


## BLEU (Bilingual Evaluation Understudy)
![[Pasted image 20240506162643.png]]BLEU’s output is always a number between 0 and 1. This value indicates how similar the candidate text is to the reference texts, with values closer to 1 representing more similar texts. Few human translations will attain a score of 1, since this would indicate that the candidate is identical to one of the reference translations. For this reason, it is not necessary to attain a score of 1. Because there are more opportunities to match, adding additional reference translations will increase the BLEU score.

This metric has multiple known limitations:

- BLEU compares overlap in tokens from the predictions and references, instead of comparing meaning. This can lead to discrepancies between BLEU scores and human ratings.
- Shorter predicted translations achieve higher scores than longer ones, simply due to how the score is calculated. A brevity penalty is introduced to attempt to counteract this.
- BLEU scores are not comparable across different datasets, nor are they comparable across different languages.
- BLEU scores can vary greatly depending on which parameters are used to generate the scores, especially when different tokenization and normalization techniques are used. It is therefore not possible to compare BLEU scores generated using different parameters, or when these parameters are unknown

## ROUGE (Recall-Oriented Understudy for Gisting Evaluation)
ROUGE score is a set of metrics commonly used for text summarization tasks, where the goal is to automatically generate a concise summary of a longer text. ROUGE was designed to evaluate the quality of machine-generated summaries by comparing them to reference summaries provided by humans.

ROUGE score measures the similarity between the machine-generated summary and the reference summaries using overlapping n-grams, word sequences that appear in both the machine-generated summary and the reference summaries. The most common n-grams used are unigrams, bigrams, and trigrams. ROUGE score calculates the recall of n-grams in the machine-generated summary by comparing them to the reference summaries.

ROUGE-1 precision can be computed as the ratio of the number of unigrams in _C_ that appear also in _R_ (that are the words “the”, “cat”, and “the”), over the number of unigrams in _C_.

```
ROUGE-1 precision = 3/5 = 0.6
```
ROUGE-1 recall can be computed as the ratio of the number of unigrams in _R_ that appear also in _C_ (that are the words “the”, “cat”, and “the”), over the number of unigrams in _R_.
```
ROUGE-1 recall = 3/6 = 0.5
```
Then, ROUGE-1 F1-score can be directly obtained from the ROUGE-1 precision and recall using the standard F1-score formula.
```
ROUGE-1 F1-score = 2 * (precision * recall) / (precision + recall) = 0.54
```

- _Pros_: it correlates positively with human evaluation, it’s inexpensive to compute and language-independent.
- _Cons_: ROUGE does not manage different words that have the same meaning, as it measures syntactical matches rather than semantics.

- BLEU focuses on precision: how much the words (and/or n-grams) in the candidate model outputs appear in the human reference.
- ==ROUGE focuses on recall: how much the words (and/or n-grams) in the human references appear in the candidate model outputs.==

## Levenshtein distance
 ![[Pasted image 20240506164256.png]]
## Perplexity

Before diving in, we should note that the metric applies specifically to classical language models (sometimes called autoregressive or causal language models) and is not well defined for masked language models like BERT
![[Pasted image 20240506165055.png]]

The perplexity metric in NLP is a way to capture the degree of ‘uncertainty’ a model has in predicting (i.e. assigning probabilities to) text.

## Accuracy, precision и recall

Перед переходом к самим метрикам необходимо ввести важную концепцию для описания этих метрик в терминах ошибок классификации — _confusion matrix_ (матрица ошибок).  
Допустим, что у нас есть два класса и алгоритм, предсказывающий принадлежность каждого объекта одному из классов, тогда матрица ошибок классификации будет выглядеть следующим образом:

![[Pasted image 20240506170644.png]]Для оценки качества работы алгоритма на каждом из классов по отдельности введем метрики precision (точность) и recall (полнота).
![[Pasted image 20240506170759.png]]
Precision можно интерпретировать как долю объектов, названных классификатором положительными и при этом действительно являющимися положительными, а recall показывает, какую долю объектов положительного класса из всех объектов положительного класса нашел алгоритм.

Именно введение precision не позволяет нам записывать все объекты в один класс, так как в этом случае мы получаем рост уровня False Positive. Recall демонстрирует способность алгоритма обнаруживать данный класс вообще, а precision — способность отличать этот класс от других классов.
```
Как мы отмечали ранее, ошибки классификации бывают двух видов: False Positive и False Negative. В статистике первый вид ошибок называют ошибкой I-го рода, а второй — ошибкой II-го рода.
```
Существует несколько различных способов объединить precision и recall в агрегированный критерий качества. F-мера (в общем случае ![$\ F_\beta$](https://habrastorage.org/getpro/habr/post_images/7c3/bf7/0f2/7c3bf70f29a2fa3bc5f72bd3dcf3a579.svg)) — среднее гармоническое precision и recall :
![[Pasted image 20240506171027.png]]
![$\beta$](https://habrastorage.org/getpro/habr/post_images/f39/05b/5cf/f3905b5cfab08d98b2d380d5ea75c66c.svg)в данном случае определяет вес точности в метрике, и при ![$\beta = 1$](https://habrastorage.org/getpro/habr/post_images/a66/57a/5ad/a6657a5ad779fd051b6a6b0fe3464c51.svg) это среднее гармоническое (с множителем 2, чтобы в случае precision = 1 и recall = 1 иметь ![$\ F_1 = 1$](https://habrastorage.org/getpro/habr/post_images/8eb/d7a/a05/8ebd7aa050e5cab1d1d63f1eb7f1e366.svg))  
F-мера достигает максимума при полноте и точности, равными единице, и близка к нулю, если один из аргументов близок к нулю.  
В sklearn есть удобная функция _metrics.classification_report_, возвращающая recall, precision и F-меру для каждого из классов, а также количество экземпляров каждого класса.
## AUC-ROC and AUC-PR
Одним из способов оценить модель в целом, не привязываясь к конкретному порогу, является AUC-ROC (или ROC AUC) — площадь (_A_rea _U_nder _C_urve) под кривой ошибок (_R_eceiver _O_perating _C_haracteristic curve ). Данная кривая представляет из себя линию от (0,0) до (1,1) в координатах True Positive Rate (TPR) и False Positive Rate (FPR)
![[Pasted image 20240506171407.png]]
TPR нам уже известна, это полнота, а FPR показывает, какую долю из объектов negative класса алгоритм предсказал неверно. В идеальном случае, когда классификатор не делает ошибок (FPR = 0, TPR = 1) мы получим площадь под кривой, равную единице; в противном случае, когда классификатор случайно выдает вероятности классов, AUC-ROC будет стремиться к 0.5, так как классификатор будет выдавать одинаковое количество TP и FP.
Каждая точка на графике соответствует выбору некоторого порога. Площадь под кривой в данном случае показывает качество алгоритма (больше — лучше), кроме этого, важной является крутизна самой кривой — мы хотим максимизировать TPR, минимизируя FPR, а значит, наша кривая в идеале должна стремиться к точке (0,1).

Критерий AUC-ROC устойчив к несбалансированным классам (спойлер: увы, не всё так однозначно) и может быть интерпретирован как вероятность того, что случайно выбранный positive объект будет проранжирован классификатором выше (будет иметь более высокую вероятность быть positive), чем случайно выбранный negative объект.
## Logistic loss
![[Pasted image 20240506172225.png]]