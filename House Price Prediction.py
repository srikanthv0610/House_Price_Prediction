#Predicting House Prices Using Linear Regression


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
import seaborn as sns

def main():
    #Importing data
    houses = pd.read_csv('AmesHousing.tsv', delimiter = '\t')

    print(np.shape(houses))
    #>>(2930 x 82)

    print(houses.head(5))


    def transform_features(df):  # Preprocessing

        ### Feature Transformation
        #Any column with more than 15% missing values will be dropped.
        #For text columns in particular, we'll drop columns with any missing values at all.
        #For numeric columns, we'll impute missing values as being the average of that column.

        #Removing columns with more than% missing values:
        df_null_count = df.isnull().sum()
        #Since there are 2930 rows in houses_2, we'll remove columns where there are more than 439 missing values.
        df_null_count = df_null_count[df_null_count < 440]
        df_filtered = df[df_null_count.index]
        print(df_filtered.dtypes)

        # we're now down to 76 columns. Next, we'll handle numeric columns with missing values.

        #Modifying Numeric Columns
        numeric_filtered = df_filtered.select_dtypes(include=['float', 'int64'])
        numeric_filtered = numeric_filtered.fillna(numeric_filtered.mean())
        # Using .sum() twice to tally up null values in the entire dataframe.
        numeric_filtered.isnull().sum().sum()

        # Now, we'll examine the text columns more closely and drop columns with any missing values at all.

        #Modifying Text Columns
        text_filtered = df_filtered.select_dtypes(include=['object'])
        text_filtered = text_filtered.dropna(axis=1)

        #Verify that there aren't any missing values: This should return a 0 ideally.
        print(text_filtered.isnull().sum().sum())

        #Concatenating Numeric and Text columns
        df_new = pd.concat([text_filtered, numeric_filtered], axis=1)
        #Creating New Features

        #Using existing coulumn: Yr Sold, Yr Built and Year Remod/Add we can create new variables:
        #The years until the house was sold
        #The years untle a house was remodelled

        df_new['Years Before Sale'] = df_new['Yr Sold'] - df_new['Year Built']
        df_new['Years Since Remod'] = df_new['Year Remod/Add'] - df_new['Year Built']

        #Removing recursive colums: Yrs Before Sale and Yrs Since Remodelled
        #Since the year a house was bought or remodelled cannot be before the year it was built,
        df_new = df_new[df_new['Years Before Sale'] >= 0]
        df_new = df_new[df_new['Years Since Remod'] >= 0]

        print(df_new.shape)
        #>> (2928, 67)

        ###Dropping columns that aren't needed

        #Order and PID columns do not give any significant information so we drop them:
        df_new = df_new.drop(["PID", "Order"], axis=1)

        #Sale Condition, Mo Sold, Sale Type and Yr Sold coulumns leak data about the final stage
        df_new = df_new.drop(["Mo Sold", "Sale Condition", "Sale Type", "Yr Sold"], axis=1)
        print(df_new.shape)
        # >> (2928, 61)
        return df_new


    def select_features(df):  #Feature Selection
        df_numeric = df.select_dtypes(include=['int64', 'float'])
        df_corr = df_numeric.corr()['SalePrice'].abs().sort_values()


        #Selecting numeric features that correlate strongly with target column
        df_numeric.corr()['SalePrice'].sort_values()
        # absolute should be applied, because we are interested in the strength of correlation
        df_corr = df_numeric.corr()['SalePrice'].abs().sort_values()
        print(df_corr)

        #We'll include only columns with a correlation greater than 0.4:
        strong_corr = df_corr[df_corr > 0.4]
        new_df = df[strong_corr.index]

        #Identifying collinearity between
        sns.heatmap(new_df.corr())
        plt.show()

        #With the help of heatmap we know the variables that are strongly correlated and thereby can be removed
        new_df = new_df.drop(['Garage Cars', 'TotRms AbvGrd'], axis=1)


        return new_df

    def train_and_test(df, k=0):

        # Identifying numeric columns, and setting them as features:
        numeric_df = df.select_dtypes(include=['integer', 'float'])
        features = numeric_df.columns.drop("SalePrice")

        # Instantiating the model:
        lr = linear_model.LinearRegression()

        if k == 0:
            # This is simply Holdout Validation
            train = df.iloc[:1460]
            test = df.iloc[1460:]

            # Training:
            lr.fit(train[features], train['SalePrice'])

            # Predicting:
            predictions = lr.predict(test[features])

            # RMSE:
            rmse = mean_squared_error(predictions, test['SalePrice']) ** 0.5

            return rmse

        elif k == 1:

            # This is Simple Cross Validation, which means we do the test twice,
            # the second time being with the train and test sets swapped.

            # First let's shuffle the dataframe:
            shuffled_df = df.sample(frac=1)  # frac = 1 means returning the whole dataframe in a shuffled manner

            # Splitting into two sets:
            train = shuffled_df[:1460]
            test = shuffled_df[1460:]

            # Training on train set, testing on test set.
            lr.fit(train[features], train["SalePrice"])
            predictions_one = lr.predict(test[features])

            mse_one = mean_squared_error(test["SalePrice"], predictions_one)
            rmse_one = np.sqrt(mse_one)

            # Here we Swap - Training on test set, testing on train set.
            lr.fit(test[features], test["SalePrice"])
            predictions_two = lr.predict(train[features])

            mse_two = mean_squared_error(train["SalePrice"], predictions_two)
            rmse_two = np.sqrt(mse_two)

            print(rmse_one)
            print(rmse_two)

            # Calculating and returning average rmse:
            avg_rmse = np.mean([rmse_one, rmse_two])
            return avg_rmse

        else:  # k-fold cross validation, where k represents number of splits
            kf = KFold(n_splits=k, shuffle=True)
            rmse_vals = []

            # Split dataframe and iterate over each train and test set.
            for train_index, test_index, in kf.split(df):
                train = df.iloc[train_index]
                test = df.iloc[test_index]
                lr.fit(train[features], train["SalePrice"])
                predictions = lr.predict(test[features])
                rmse = (mean_squared_error(test["SalePrice"], predictions)) ** 0.5
                print(rmse)
                rmse_vals.append(rmse)

                plt.figure()
                plt.scatter(test["SalePrice"], predictions)
                plt.xlabel('Y Test')
                plt.ylabel('Predicted Y')
                plt.show()

            avg_rmse = np.mean(rmse_vals)
            return avg_rmse

    transform_df = transform_features(houses)
    filtered_df = select_features(transform_df)
    rmse = train_and_test(filtered_df, k=5)  # Specifying k = 5
    print(rmse)

    #We observe that we managed to decrease the rmse to the 32000-34000 range with k-fold cross validation.


if __name__ == '__main__':
    main()