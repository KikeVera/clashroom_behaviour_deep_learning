import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.optimizers import Adam

from keras.layers import Dense
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler


def generate_synthetic_data(dataframe: pd.DataFrame):
    # Crear modelo discriminador
    scaler = MinMaxScaler()
    dataframe = dataframe.drop("action", axis=1)
    X_normalized = pd.DataFrame(scaler.fit_transform(dataframe), columns=dataframe.columns)
    discriminator = Sequential()
    discriminator.add(Dense(2000, input_dim=X_normalized.shape[1], activation='relu')),
    discriminator.add(Dense(2000, input_dim=X_normalized.shape[1], activation='relu')),
    discriminator.add(Dense(2000, input_dim=X_normalized.shape[1], activation='relu')),
    discriminator.add(Dense(2000, input_dim=X_normalized.shape[1], activation='relu')),
    discriminator.add(Dense(2000, input_dim=X_normalized.shape[1], activation='relu')),
    discriminator.add(Dense(2000, input_dim=X_normalized.shape[1], activation='relu')),



    discriminator.add(Dense(1, activation='sigmoid'))
    discriminator.trainable = False
    discriminator.compile(loss='mean_squared_error', optimizer='adam')

    # Crear modelo generador
    generator = Sequential()
    generator.add(Dense(1000, input_dim=X_normalized.shape[1], activation='relu')),
    generator.add(Dense(1000, input_dim=X_normalized.shape[1], activation='relu')),






    generator.add(Dense(X_normalized.shape[1], activation='sigmoid'))

    # Congelar el discriminador durante el entrenamiento del modelo GAN
    discriminator.trainable = False

    # Construir el modelo GAN
    gan = Sequential([generator, discriminator])
    gan.compile(loss='mean_squared_error', optimizer='adam')

    # Entrenamiento de la GAN (esto es un ejemplo, ajusta según tus datos)
    epochs = 200
    batch_size = 32

    for epoch in range(epochs):
        # Generar datos aleatorios en el espacio latente para el generador
        noise = np.random.normal(0, 1, (batch_size, X_normalized.shape[1]))
        generated_data_batch = generator.predict(noise)

        # Seleccionar un lote aleatorio de datos reales
        idx = np.random.randint(0, X_normalized.shape[0], batch_size)
        real_data_batch = X_normalized.loc[idx]

        # Etiquetas para datos reales y generados
        real_labels = np.ones((batch_size, 1))
        generated_labels = np.zeros((batch_size, 1))

        # Entrenar el discriminador
        d_loss_real = discriminator.train_on_batch(real_data_batch, real_labels)
        d_loss_generated = discriminator.train_on_batch(generated_data_batch, generated_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_generated)

        # Entrenar el generador
        noise = np.random.normal(0, 1, (batch_size, X_normalized.shape[1]))
        g_loss = gan.train_on_batch(noise, real_labels)

        # Mostrar progreso cada cierto número de epochs
        if epoch % 50 == 0:
            print(f"Epoch: {epoch}, Discriminator Loss: {d_loss}, Generator Loss: {g_loss}")

    # Generar datos sintéticos
    num_samples = 2000
    noise = np.random.normal(0, 1, (num_samples, X_normalized.shape[1]))
    synthetic_data = generator.predict(noise)

    synthetic_data = pd.DataFrame(scaler.inverse_transform(synthetic_data), columns=dataframe.columns)
    return synthetic_data

    # Desnormalizar los datos si es necesario
    # synthetic_data = synthetic_data * (X.max() - X.min()) + X.min()

    # Visualización de los datos sintéticos generados
    # plt.scatter(synthetic_data[:, 0], synthetic_data[:, 1], c='blue', label='Synthetic Data')
    # plt.legend()
    # plt.title('Datos Sintéticos Generados por GAN')
    # plt.xlabel('Feature 1')
    # plt.ylabel('Feature 2')
    # plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    prepared_df = pd.read_csv("data/datasets/standard/standard_dataset.csv")
    for action in prepared_df["action"].unique():
        action_df = prepared_df[prepared_df["action"] == action]
        path = "data/datasets/synthetic_data/"
        if not os.path.exists(path):
            os.makedirs(path)
        synthetic_df = generate_synthetic_data(action_df)
        synthetic_df["action"] = action
        synthetic_df.to_csv(path+action+".csv", index =False)


