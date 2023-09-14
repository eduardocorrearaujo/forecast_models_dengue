import pickle 
from preprocessing import get_ml_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from crepes import ConformalRegressor
from crepes.extras import DifficultyEstimator
from plots import plot_prediction


def make_pred(city, ini_date, end_train_date, end_date, ratio, filename, label, doenca= 'dengue', predict_n = 4, look_back=4): 

    X_data, X_train, targets, target = get_ml_data(city, ini_date = ini_date, end_train_date = end_train_date, end_date = end_date, 
                                        ratio = ratio , predict_n = predict_n, look_back = look_back, filename = filename)

    d = 4 # o alvo é o número de casos na semana daqui a 4 semanas
    y_train = targets[d][:len(X_train)]
    
    y_data = targets[d]

    model =  RandomForestRegressor(oob_score=True) 

    X_prop_train, X_cal, y_prop_train, y_cal = train_test_split(X_train, y_train,
                                                                test_size=0.25)

    model.fit(X_prop_train, y_prop_train)

    # create a conformal regression instance 
    cr_std = ConformalRegressor()

    y_hat_cal = model.predict(X_cal)

    residuals_cal = y_cal - y_hat_cal

    cr_std.fit(residuals_cal)

    X_test = X_data[X_train.shape[0]:len(y_data)]
    y_test = y_data[X_train.shape[0]:]

    y_hat_test = model.predict(X_data[:len(targets[d])])

    #intervals = cr_std.predict(y_hat_test, confidence=0.95, y_min = 0)

    de_knn = DifficultyEstimator()

    de_knn.fit(X=X_prop_train, scaler=True)

    sigmas_cal_knn_dist = de_knn.apply(X_cal)

    cr_norm_knn_dist = ConformalRegressor()

    cr_norm_knn_dist.fit(residuals_cal, sigmas=sigmas_cal_knn_dist)

    sigmas_test_knn_dist = de_knn.apply(X_data[:len(targets[d])])

    intervals_norm_knn_dist = cr_norm_knn_dist.predict(y_hat_test, 
                                                   sigmas=sigmas_test_knn_dist,
                                                   y_min=0)
    

    with open(f'./predictions/rf_{city}_{doenca}_{label}_predictions.pkl', 'wb') as f:
        pickle.dump({'target':targets[4].values,'dates': X_data.index[4:], 'preds': y_hat_test,
                      'preds25': intervals_norm_knn_dist[:,0],
                    'preds975': intervals_norm_knn_dist[:,1], 'train_size': len(X_train)
                    }, f)
    
    plot_prediction(city, X_data.index[4:], 
                    targets[4].values,
                    y_hat_test,
                    intervals_norm_knn_dist[:,0],
                    intervals_norm_knn_dist[:,1],
                     len(X_train), doenca, label)   
    






