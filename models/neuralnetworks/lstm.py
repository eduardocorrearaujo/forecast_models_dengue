import pickle
import numpy as np
import pandas as pd
from time import time 
import tensorflow as tf 
import tensorflow.keras as keras
from preprocessing import get_nn_data
from tensorflow.keras.layers import LSTM
from plots_lstm import plot_train_test
from tensorflow.keras.utils import plot_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout, Bidirectional
from sklearn.metrics import mean_absolute_error, explained_variance_score, mean_squared_error, mean_squared_log_error, median_absolute_error, r2_score
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

MAIN_FOLDER = '.'

def evaluate(model, Xdata, batch_size,  uncertainty=True):
    """
    Function to make the predictions of the model trained 
    :param model: Trained lstm model 
    :param Xdata: Array with data
    :param uncertainty: boolean. If True multiples predictions are returned. Otherwise, just
                        one prediction is returned. 
    """
    if uncertainty:
        predicted = np.stack([model.predict(Xdata, batch_size=batch_size, verbose=0) for i in range(100)], axis=2)
    else:
        predicted = model.predict(Xdata, batch_size=batch_size, verbose=0)
    return predicted

def calculate_metrics(pred, y_true, factor):
    """
    Function to compute some metrics given a vector with the predictions and the truth values
    :param pred: array with the predictions. 
    :param y_true: array with the truth values 
    :param factor:
    """
    metrics = pd.DataFrame(
        index=(
            "mean_absolute_error",
            "explained_variance_score",
            "mean_squared_error",
            "mean_squared_log_error",
            "median_absolute_error",
            "r2_score",
        )
    )
    for col in range(pred.shape[1]):
        y = y_true[:, col] * factor
        p = pred[:, col] * factor
        l = [
            mean_absolute_error(y, p),
            explained_variance_score(y, p),
            mean_squared_error(y, p),
            mean_squared_log_error(y, p),
            median_absolute_error(y, p),
            r2_score(y, p),
        ]
        metrics[col] = l
    return metrics


def build_model(l1=1e-5, l2=1e-5, hidden=8, features=100, predict_n=4, look_back=4, batch_size=1, loss = 'msle', lr = 0.005, f_act_1 = 'tanh', f_act_2 = 'tanh'):
    """
    Builds and returns the LSTM model with the parameters given
    :param hidden: int.number of hidden nodes
    :param features: int. Number of the features used to train the model
    :param predict_n: int. Number of observations that will be forecast (multi-step forecast)
    :param look_back: int. Number of time-steps to look back before predicting
    :param batch_size: int. batch size for batch training
    :param loss: string or function. Loss function to be used in the training process. 
    :return:
    """
    
 
    
    inp = keras.Input(
        shape=(look_back, features),
        # batch_shape=(batch_size, look_back, features)
    )
    
    x = Bidirectional(LSTM(
        hidden,
        input_shape=(look_back, features),
        stateful=False,
        batch_input_shape=(batch_size, look_back, features),
        return_sequences=True,
        kernel_regularizer=regularizers.L1L2(l1=l1, l2=l2),
        bias_regularizer=regularizers.L2(l2),
        #activity_regularizer=regularizers.L2(1e-5), 
        activation=f_act_1,
        dropout=0.1,
        recurrent_dropout=0.1,
        implementation=2,
        unit_forget_bias=True,
    ), merge_mode = 'ave', name = 'bidirectional_1')(inp, training=True)     

    x = Dropout(0.2, name='dropout_1')(x, training=True)
    
    x = LSTM(
        hidden,
        input_shape=(look_back, features),
        stateful=False,
        batch_input_shape=(batch_size, look_back, features),
        return_sequences= True,
        kernel_regularizer=regularizers.L1L2(l1=l1, l2=l2),
        bias_regularizer=regularizers.L2(l2),
        #activity_regularizer=regularizers.L2(1e-5), 
        activation=f_act_2,
        dropout=0.1,
        recurrent_dropout=0.1,
        implementation=2,
        unit_forget_bias=True, name='lstm_1'
    )(x, training=True)

    x = Dropout(0.2, name='dropout_2')(x, training=True)
    
    x = LSTM(
        hidden,
        input_shape=(look_back, features),
        stateful=False,
        batch_input_shape=(batch_size, look_back, features),
        return_sequences= False,
        kernel_regularizer=regularizers.L1L2(l1=l1, l2=l2),
        bias_regularizer=regularizers.L2(l2),
        #activity_regularizer=regularizers.L2(1e-5), 
        activation=f_act_2, 
        dropout=0.1,
        recurrent_dropout=0.1,
        implementation=2,
        unit_forget_bias=True, name='lstm_2'
    )(x, training=True)


    x = Dropout(0.2, name = 'dropout_3')(x, training=True)

    out = Dense(
        predict_n,
        activation="relu",
        activity_regularizer=regularizers.L2(l2), 
        kernel_initializer="random_uniform",
        bias_initializer="zeros",
        name = 'dense'
    )(x)
    model = keras.Model(inp, out)

    start = time()
    optimizer = keras.optimizers.Adam(learning_rate = lr)
    #optimizer = tf.keras.optimizers.experimental.SGD(learning_rate=0.0001)

    model.compile(loss = loss, optimizer=optimizer, metrics=["accuracy", "mape", "mse"])
    print("Compilation Time : ", time() - start)
    plot_model(model, to_file="LSTM_model.png")
    print(model.summary())
    return model


def train(model, X_train, Y_train, label, batch_size=1, epochs=10, geocode=None, overwrite=True, validation_split = 0.25, patience = 50, monitor = 'val_loss'):
    """
    Train the lstm model 
    :param model: LSTM model compiled and created with the build_model function 
    :param X_train: array. Arrays with the features to train the model 
    :param Y_train: array. Arrays with the target to train the model
    :param label: string. Name to be used to save the model
    :param batch_size: int. batch size for batch training
    :param epochs: int.  Number of epochs used in the train 
    :param geocode: int. Analogous to city (IBGE code), it will be used in the name of the saved model
    :param overwrite: boolean. If true we overwrite a saved model with the same name. 
    :param validation_split: float. The slice of the training data that will be use to evaluate the model 
    """
    
    TB_callback = TensorBoard(
        log_dir="./tensorboard",
        histogram_freq=0,
        write_graph=True,
        write_images=True,
        # embeddings_freq=10
    )

    seed = 7 

    if validation_split > 0.0:
        X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, test_size=validation_split, random_state=seed)
        

        hist = model.fit(
            X_train,
            Y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_test,Y_test), 
            verbose=1,
            callbacks=[TB_callback, EarlyStopping(monitor = monitor, patience=patience)]
        )
        
        model.save(f"{MAIN_FOLDER}/saved_models/trained_{geocode}_model_{label}.h5", overwrite=overwrite)


        pred_train = np.percentile(np.stack([model.predict(X_train, batch_size=batch_size, verbose=0) for i in range(100)], axis=2), 50, axis=2)
        pred_test = np.percentile(np.stack([model.predict(X_test, batch_size=batch_size, verbose=0) for i in range(100)], axis=2), 50, axis=2)

        
        metrics_train = calculate_metrics(pred_train, Y_train, 1)

        metrics_val = calculate_metrics(pred_test, Y_test, 1)
    
    else: 

        hist = model.fit(
            X_train,
            Y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            callbacks=[TB_callback, EarlyStopping(monitor = 'loss', patience=patience)]
        )
        
        model.save(f"{MAIN_FOLDER}/saved_models/lstm/trained_{geocode}_model_{label}.h5", overwrite=overwrite)


        pred_train = np.percentile(np.stack([model.predict(X_train, batch_size=batch_size, verbose=0) for i in range(100)], axis=2), 50, axis=2)
        
        metrics_train = calculate_metrics(pred_train, Y_train, 1)

        metrics_val = pd.DataFrame()

    return model, hist, metrics_train, metrics_val 


def make_pred(model, city, doenca,  epochs, ini_date = None, end_train_date = None, 
                 end_date = None,ratio= 0.75,
                 predict_n = 4, look_back =  4, batch_size = 4, 
                  label = 'model', filename = None):
    """
    The parameters ended with the word `date` are used to apply the model in different time periods. 
    :param model: tensorflow model. 
    :param city: int. IBGE code of the city. 
    :param doenca: string. Is used to name the trained model. 
    :param epochs: int. 
    :param ratio: float. Percentage of the data used to train and test steps. 
    :param ini_date: string or None. Determines after which day the data will be used 
    :param end_train_date: string or None. Determines the last day used to train the data. 
                         If not None the parameter ratio is unused. 
    :param end_date: string or None. Determines the last day used 
    :param predict_n: int. Number of observations that it will be forecasted 
    :param look_back: int. Number of last observations used as input 
    :param label: string.
    :param filename: string. Path to the data used to train and evaluate the model. 
    """

    df,factor,  X_train, Y_train, X_pred, Y_pred = get_nn_data(city, ini_date = ini_date, 
                                                     end_date = end_date, end_train_date = end_train_date,
                                                    ratio = ratio, look_back = look_back,
                                                    predict_n = predict_n, filename = filename)
   
    model, hist, m_train, m_val =  train(model, X_train, Y_train, label = label, batch_size=batch_size, epochs=epochs, geocode=city, overwrite=True, validation_split = 0.25, monitor = 'val_loss')
   
    pred = evaluate(model, X_pred, batch_size)

    df_pred = pd.DataFrame(np.percentile(pred, 50, axis=2))
    df_pred25 = pd.DataFrame(np.percentile(pred, 2.5, axis=2))
    df_pred975 = pd.DataFrame(np.percentile(pred, 97.5, axis=2))
    
    with open(f'{MAIN_FOLDER}/predictions/lstm_{city}_{doenca}_{label}.pkl', 'wb') as f:
        pickle.dump({'xdata': X_train, 'indice': list(df.index)  , 'target': Y_pred,  'pred': df_pred, 'ub': df_pred975,  'lb':df_pred25,
                    'factor': factor, 'city': city}, f)

    indice = list(df.index)
    indice = [i.date() for i in indice] 

    plot_train_test(indice,  Y_pred, factor, df_pred, df_pred25, df_pred975, len(X_train), city)                    

    metrics = calculate_metrics(np.percentile(pred, 50, axis=2), Y_pred, factor)

    return metrics, hist, m_train, m_val   

