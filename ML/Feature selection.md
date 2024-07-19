![[feature-selection-methods-1.png]]
##### SHAP values
- Provides a way to fairly distribute the contribution of each feature in a predictive model to the prediction for a specific instance

1. Define features power set
2. Determine the marginal contribution: calculate prediction with/without the feature ``` 
```
f(0) = 0 
f(A) = 0.5
f(A,B) = 0.8
f(A,C) = 0.7
f(B) = 0.3
f(B,C) = 0.6
f(C) = 0.2
f(A,B,C)=1.0

Marginal contribution of A is: 0.5
For B -> 0.5 
For C -> 0.5
 ```
 3. Shapley values ```
``` 
(0.5+0.8+0.7+1.0)/6 = 0.5 for value A 
```
###### Example:
```
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_train)

vals=np.mean([np.mean(np.abs(class_shap_values), axis=0) for class_shap_values in shap_values], axis=0)
feature_importance = pd.DataFrame(list(zip(X_train.columns,vals)),columns=['col_name','feature_importance_vals'])
feature_importance.sort_values(by=['feature_importance_vals'],ascending=False,inplace=True)
top_1000=list(feature_importance.head(1000)['col_name'])
```
##### Phik values
1. Phi-k coeff -> It's a measure of associating between two categorical variables
2. Calculation -> Uses an idea of information gain and is computed by: 
$$
PhiK(X,Y)= \sqrt{\dfrac{x_2(X,Y)}{N*min(K_x-1,K_y-1)}}
$$
##### LASSO: Least absolute shrinkage and selection operator

- Regularization technique used to prevent overfitting and encourage sparse model by adding a penalty term to the sum of squared coefficients
```
L1 = lambda * sum(abs(w_j))
```
- LASSO minimization problem is:
```
minimize(sum(y_i - y_hat)**2 + lambda * sum(abs(w_j)))
```
- Magnitudes indicate influence

#### Filter methods-metrics
##### Pearson's correlation

$$
\dfrac{covariance(X,Y)}{\sqrt{var(X)}*\sqrt{var(Y)}}
$$
$$
covariance(X,Y)=\dfrac{\sum_{i=1}^{n}(x_i-\hat{x})(y_i-\hat{y})}{N-1}
$$
##### Chi-squared 

$$
\chi^2 = \sum \frac{(O - E)^2}{E}
$$

##### Mutual information
$$
I(X;Y) = \sum_{y \in Y} \sum_{x \in X} p(x,y) \log\left(\frac{p(x,y)}{p(x)p(y)}\right) 
$$

