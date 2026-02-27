# Prophet: Pronóstico de Series de Tiempo
# Autor: Luis Rojas - 20202020242
# Curso: Probabilidad y Estadística - Grupo 84

import numpy as np
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt

# Genera datos simulados (400 días históricos)
np.random.seed(42)
n_dias = 400
fechas = pd.date_range(start='2020-01-01', periods=n_dias, freq='D')

# Componentes del modelo
tendencia = 0.05 * np.arange(n_dias)
estacionalidad = 3 * np.sin(2 * np.pi * np.arange(n_dias) / 7)
ruido = np.random.normal(0, 2, n_dias)
y = 50 + tendencia + estacionalidad + ruido

df = pd.DataFrame({'ds': fechas, 'y': y})

# Crea y entrena el modelo
m = Prophet(weekly_seasonality=True, yearly_seasonality=False)
m.fit(df)

# Pronosticar 60 días
future = m.make_future_dataframe(periods=60, freq='D')
forecast = m.predict(future)

# Visualizar
m.plot(forecast)
plt.title('Pronóstico de Demanda Diaria')
plt.show()

m.plot_components(forecast)
plt.show()
