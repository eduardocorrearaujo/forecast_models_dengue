import numpy as np 
import pandas as pd 
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize, LabelEncoder

def build_lagged_features(dt, lag=2, dropna=True):
    '''
    returns a new DataFrame to facilitate regressing over all lagged features.
    :param dt: Dataframe containing features
    :param lag: maximum lags to compute
    :param dropna: if true the initial rows containing NANs due to lagging will be dropped
    :return: Dataframe
    '''
    if type(dt) is pd.DataFrame:
        new_dict = {}
        for col_name in dt:
            new_dict[col_name] = dt[col_name]
            # create lagged Series
            for l in range(1, lag + 1):
                new_dict['%s_lag%d' % (col_name, l)] = dt[col_name].shift(l)
        res = pd.DataFrame(new_dict, index=dt.index)

    elif type(dt) is pd.Series:
        the_range = range(lag + 1)
        res = pd.concat([dt.shift(-i) for i in the_range], axis=1)
        res.columns = ['lag_%d' % i for i in the_range]
    else:
        print('Only works for DataFrame or Series')
        return None
    if dropna:
        return res.dropna()
    else:
        return res


def get_ml_data(city, ini_date, end_train_date, end_date, ratio, predict_n, look_back, filename = None):


    data = pd.read_csv(filename, index_col = 'Unnamed: 0' )
    data.index = pd.to_datetime(data.index)  

    for i in data.columns:

        if i.startswith('casos'):

            data[f'diff_{i}'] = np.concatenate( ([np.nan], np.diff(data[i], 1)))

    data = data.dropna()

    target = f'casos_{city}'

    data_lag = build_lagged_features(data, look_back)
    data_lag.dropna()
    
    if ini_date != None:
        data_lag = data_lag.loc[ini_date: ]

    if end_date != None:
        data_lag = data_lag.loc[:end_date]

    X_data = data_lag.copy()

    # Let's remove the features not lagged from X_data to be sure that we are using just
    # past information in the forecast
    drop_columns = []
    for i in X_data.columns:
        if 'lag' not in i: 
            drop_columns.append(i)

    X_data = X_data.drop(drop_columns, axis=1)

    targets = {}
    for d in range(1, predict_n + 1):
        targets[d] = data_lag[target].shift(-(d))[:-(d)]

    if end_train_date == None: 
        X_train, X_test, y_train, y_test = train_test_split(X_data, data_lag[target],
                                                        train_size = ratio, test_size = 1 -ratio, shuffle=False)
    
    else: 
        X_train = X_data.loc[:end_train_date]

    return X_data, X_train, targets, data_lag[target] 









