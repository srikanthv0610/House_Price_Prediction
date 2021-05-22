# House_Price_Prediction
Using the dataset compiled by Dean De Cock. 
Applying Feature Transformation, Feature Selection and K-fold Cross Validation.
We use Linear Regression to Predict the House Prices.

#Feature Transformation
* Check the columns with more than 15% missing values and drop them.
* For text columns, we drop columns with any missing values at all.
* For numeric columns, we fill the missing values by the mean(average) of that column.
* Add new features based on the existing columns.
* Drop columns that weren't significant, or which leaked data on the sale.

#Feature Selection
* Identify numeric columns that correlated strongly with target columns, and selected those with strong correlations (> 0.4)
* Generate Heatmap for identifying collinearity between columns and drop variables that are strongly correlated.

![Heatmap](https://github.com/srikanthv0610/House_Price_Prediction/blob/main/plots/Heatmap_Collinearity.png)
Format: ![Alt Text](url)
