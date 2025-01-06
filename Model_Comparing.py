import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

def KNeighborsRegressor_Model(Stock_Data, Name):
    Stock = Stock_Data.loc[Stock_Data['Name'] == Name]
    X = Stock[['Open', 'High', 'Low', 'Volume']]
    Y = Stock['Close']

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    Model = KNeighborsRegressor(n_neighbors=850, weights='uniform').fit(X_train, Y_train)

    Predictions = Model.predict(X_test)
    Test_Dates = Stock_Data.loc[Y_test.index, 'Date']
    Results = pd.DataFrame({
        'Date': Test_Dates,
        'Actual': Y_test.values,
        'Predicted': Predictions
    })

    Results['Trend Prediction'] = np.where(Results['Predicted'].diff() > 0, 'Increase', 'Decrease')
    Results['Trend Actual'] = np.where(Results['Actual'].diff() > 0, 'Increase', 'Decrease')

    Accuracy = (Results['Trend Prediction'] == Results['Trend Actual']).mean()
    print(f'Accuracy of trend prediction: {Accuracy:.2%}')

    return Results.sort_values(by='Date').reset_index(drop=True), Model


