from src.utils.utils import *
import random, matplotlib.pyplot as plt, pandas as pd, seaborn as sns

folders = ['src/experiments/runet', 'src/experiments/runetv2']

routes = get_files_path(random.choice(folders))

x = [extract_history_data(load_history(route)) for route in routes]

runet_y = [extract_history_loss(load_history(route)) for route in get_files_path(folders[0])]
runetv2_y = [extract_history_loss(load_history(route)) for route in get_files_path(folders[1])]

runet_y2 = [extract_history_time(load_history(route)) for route in get_files_path(folders[0])]
runetv2_y2 = [extract_history_time(load_history(route)) for route in get_files_path(folders[1])]


# Crear el DataFrame (el código para crear las series de datos se asume que ya lo tienes)
df = pd.DataFrame({
    'Params': x,
    'RUnet loss': runet_y,
    'RUnetv2 loss': runetv2_y,
    'RUnet time': runet_y2,
    'RUnetv2 time': runetv2_y2
})


# Crear la figura y los ejes
fig, ax1 = plt.subplots(figsize=(14, 8))

# Configurar el primer eje Y para las pérdidas
sns.lineplot(x='Params', y='RUnet loss', data=df, ax=ax1, label='RUnet Loss', color='#1f77b4', linewidth=2.5, marker='o')
sns.lineplot(x='Params', y='RUnetv2 loss', data=df, ax=ax1, label='RUnetv2 Loss', color='#ff7f0e', linewidth=2.5, marker='s')

ax1.set_xlabel('Data Size', fontsize=16, weight='bold')
ax1.set_ylabel('Loss', fontsize=16, weight='bold', color='black')
ax1.tick_params(axis='y', labelcolor='black', labelsize=12)
ax1.tick_params(axis='x', labelsize=12)

# Crear el segundo eje Y para el tiempo
ax2 = ax1.twinx()
sns.lineplot(x='Params', y='RUnet time', data=df, ax=ax2, label='RUnet Time', color='#2ca02c', linewidth=2.5, linestyle='--', marker='^')
sns.lineplot(x='Params', y='RUnetv2 time', data=df, ax=ax2, label='RUnetv2 Time', color='#d62728', linewidth=2.5, linestyle='--', marker='d')

ax2.set_ylabel('Time (seconds)', fontsize=16, weight='bold', color='black')
ax2.tick_params(axis='y', labelcolor='black', labelsize=12)

# Título principal
plt.title('Data size against time and loss', fontsize=20, weight='bold')

# Ajustar leyendas para que no se sobrepongan
ax1.legend(loc='upper left', fontsize=12, frameon=True, shadow=True)
ax2.legend(loc='upper right', fontsize=12, frameon=True, shadow=True)

# Ajustar la cuadrícula para ser menos intrusiva
ax1.grid(True, linestyle='--', color='gray', alpha=0.3)  # Color más claro y menos opacidad

# Asegurar que todos los elementos se ajusten correctamente en la figura
fig.tight_layout()

# Mostrar gráfico
plt.show()