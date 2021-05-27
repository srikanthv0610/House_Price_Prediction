# House_Price_Prediction
Used the AmesHousing dataset [here](https://github.com/srikanthv0610/House_Price_Prediction/edit/main/Dataset) compiled by Dean De Cock.

Apply Feature Transformation, Feature Selection and K-fold Cross Validation.

Linear Regression to Predict the House Prices.

## Feature Transformation
* Check the columns with more than 15% missing values and drop them.
* For text columns, we drop columns with any missing values at all.
* For numeric columns, we fill the missing values by the mean(average) of that column.
* Add new features based on the existing columns.
* Drop columns that weren't significant, or which leaked data on the sale.

## Feature Selection
* Identify numeric columns that correlated strongly with target columns, and selected those with strong correlations (> 0.4)
* Generate Heatmap for identifying collinearity between columns and drop variables that are strongly correlated.

![Heatmap](https://github.com/srikanthv0610/House_Price_Prediction/blob/main/plots/Heatmap_Collinearity2.png)

## K-Fold Cross Validation
A parameter k is introduced to the defined train_test function that controls the types of cross-Validation for:
* k = 0 
* k = 1 
* k > 1

## Predict using Linear Regression
Once the Data Preprocessing and K-Fold Cross Validation is done, we use Linear Regression to predict the House Price and we compute the RMSE to evaluate the quality of Prediction:

![Lm](https://github.com/srikanthv0610/House_Price_Prediction/blob/main/plots/Figure_1.png)

```python
* from sklearn import linear_model
* RMSE = mean_squared_error(test_set, predictied_value) 

