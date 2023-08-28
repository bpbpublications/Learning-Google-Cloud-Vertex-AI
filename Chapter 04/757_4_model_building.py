import numpy as np
import pandas as pd
import pathlib
import tensorflow as tf
# Importing tensorflow 2.6
from tensorflow import keras
from tensorflow.keras import layers


def main():
    # Reading data from the gcs bucket
    dataset = pd.read_csv(r"gs://custom_model_ele_prices/electricity_prices.csv", low_memory=False)
    dataset.tail()
    BUCKET = 'gs://custom_model_ele_prices'
    dataset.isna().sum()
    dataset = dataset.dropna()
    dataset.drop(['DateTime', 'Holiday'], axis=1, inplace=True)
    cols = list(dataset.columns[dataset.dtypes.eq('object')])
    dataset[cols] = dataset[cols].apply(pd.to_numeric, errors='coerce', axis=1)
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)
    train_stats = train_dataset.describe()
    train_stats.pop("SMPEP2")
    train_stats = train_stats.transpose()
    train_labels = train_dataset.pop('SMPEP2')
    test_labels = test_dataset.pop('SMPEP2')
    normed_train_data = (train_dataset - train_dataset.mean()) / train_dataset.std()
    normed_test_data = (test_dataset - train_dataset.mean()) / train_dataset.std()

    def tf_build_model_reg():
        # model building function
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
            layers.Dense(64, activation='relu'),
            layers.Dense(128, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(1)])
        optimizer = tf.keras.optimizers.Adagrad(0.001)
        model.compile(loss='mse', optimizer=optimizer, metrics=['mae', 'mse'])
        return model

    model = tf_build_model_reg()
    epochs = 10

    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

    early_history = model.fit(normed_train_data, train_labels, epochs=epochs, validation_split=0.2,
                              callbacks=[early_stop])
    model.save(BUCKET + '/model')


if __name__ == "__main__":
    main()
