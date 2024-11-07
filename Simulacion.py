import numpy as np
import pandas as pd
import sqlite3
from numba import njit, prange

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

@njit
def combinar_v_r(v, r):
    result = np.empty((12,), dtype=np.float64)
    
    idx = 0
    for i in range(3):
        result[idx] = v[i][0]
        result[idx + 1] = v[i][1]
        idx += 2

    idx = 6
    for i in range(3):
        result[idx] = r[i][0]
        result[idx + 1] = r[i][1]
        idx += 2
    
    return result

@njit
def aceleracion(mj, mk, ri, rj, rk):
    r_ij_x = ri[0] - rj[0]
    r_ij_y = ri[1] - rj[1]
    r_ik_x = ri[0] - rk[0]
    r_ik_y = ri[1] - rk[1]

    mag_cub_ij = (r_ij_x**2 + r_ij_y**2)**1.5
    mag_cub_ik = (r_ik_x**2 + r_ik_y**2)**1.5

    ax = -(mj * r_ij_x / mag_cub_ij + mk * r_ik_x / mag_cub_ik)
    ay = -(mj * r_ij_y / mag_cub_ij + mk * r_ik_y / mag_cub_ik)

    return np.array([ax, ay], dtype=np.float64)

@njit
def calcular_paso_runge_kutta(h, r_inicial, v_inicial, m):
    a1 = np.empty((3,2), dtype=np.float64)
    a1[0] = aceleracion(m[1], m[2], r_inicial[0], r_inicial[1], r_inicial[2])
    a1[1] = aceleracion(m[2], m[0], r_inicial[1], r_inicial[2], r_inicial[0])
    a1[2] = aceleracion(m[0], m[1], r_inicial[2], r_inicial[0], r_inicial[1])
    k1_r = h * v_inicial
    k1_v = h * a1

    r_medio = r_inicial + 0.5 * k1_r
    v_medio = v_inicial + 0.5 * k1_v

    a2 = np.empty((3,2), dtype=np.float64)
    a2[0] = aceleracion(m[1], m[2], r_medio[0], r_medio[1], r_medio[2])
    a2[1] = aceleracion(m[2], m[0], r_medio[1], r_medio[2], r_medio[0])
    a2[2] = aceleracion(m[0], m[1], r_medio[2], r_medio[0], r_medio[1])
    
    k2_r = h * v_medio
    k2_v = h * a2

    r_medio = r_inicial + 0.5 * k2_r
    v_medio = v_inicial + 0.5 * k2_v

    a3 = np.empty((3,2), dtype=np.float64)
    a3[0] = aceleracion(m[1], m[2], r_medio[0], r_medio[1], r_medio[2]) 
    a3[1] = aceleracion(m[2], m[0], r_medio[1], r_medio[2], r_medio[0]) 
    a3[2] = aceleracion(m[0], m[1], r_medio[2], r_medio[0], r_medio[1])
    
    k3_r = h * v_medio
    k3_v = h * a3

    r_final = r_inicial + k3_r
    v_final = v_inicial + k3_v

    a4 = np.empty((3,2), dtype=np.float64)
    a4[0] = aceleracion(m[1], m[2], r_final[0], r_final[1], r_final[2]) 
    a4[1] = aceleracion(m[2], m[0], r_final[1], r_final[2], r_final[0])
    a4[2] = aceleracion(m[0], m[1], r_final[2], r_final[0], r_final[1])
    
    k4_r = h * v_final
    k4_v = h * a4

    r_nuevo = r_inicial + (k1_r + 2 * k2_r + 2 * k3_r + k4_r) / 6
    v_nuevo = v_inicial + (k1_v + 2 * k2_v + 2 * k3_v + k4_v) / 6

    return combinar_v_r(v_nuevo, r_nuevo)

@njit
def calcular_paso_euler(h, r_inicial, v_inicial, m):
    r = r_inicial + h * v_inicial

    a1 = aceleracion(m[1], m[2], r_inicial[0], r_inicial[1], r_inicial[2])
    a2 = aceleracion(m[2], m[0], r_inicial[1], r_inicial[2], r_inicial[0])
    a3 = aceleracion(m[0], m[1], r_inicial[2], r_inicial[0], r_inicial[1])

    v = np.empty((3, 2), dtype=np.float64)
    v[0] = v_inicial[0] + h * a1
    v[1] = v_inicial[1] + h * a2
    v[2] = v_inicial[2] + h * a3

    return combinar_v_r(v, r)

@njit
def calcular_todos_los_pasos(t_inicio, t_final, n, r_inicial, v_inicial, m):
    h = (t_final - t_inicio) / n
    resultados = np.empty((n, 12), dtype=np.float64)
    
    for i in range(n):
        # resultados[i] = calcular_paso_euler(h, r_inicial, v_inicial, m)
        resultados[i] = calcular_paso_runge_kutta(h, r_inicial, v_inicial, m)
        v_inicial = np.reshape(resultados[i][0:6], (3,2))
        r_inicial = np.reshape(resultados[i][6:12], (3,2))
    
    return resultados

@njit
def calcular_ciclos(n, limite_n, t_inicio, t_final, v_inicial, r_inicial, m):
    resultados = np.empty((limite_n, 12), dtype=np.float64)

    ciclos = int(np.ceil(n / limite_n))
    delta_t = (t_final - t_inicio) / ciclos

    temporal_t_inicios = np.arange(t_inicio, t_inicio + delta_t * ciclos, delta_t)
    start_indexes = np.arange(0, int(np.ceil(limite_n/ciclos))*ciclos, int(np.ceil(limite_n/ciclos)))

    for j in prange(ciclos):
        temporal_t_inicio = temporal_t_inicios[j]
        temporal_t_final = temporal_t_inicios[j] + delta_t
        temporal_n = limite_n if (j + 1 < ciclos) or (n % limite_n == 0) else n % limite_n

        resultados_temporales = calcular_todos_los_pasos(temporal_t_inicio, temporal_t_final, temporal_n, r_inicial, v_inicial, m)
        v_inicial = np.reshape(resultados_temporales[-1][0:6], (3,2))
        r_inicial = np.reshape(resultados_temporales[-1][6:12], (3,2))

        start_index = start_indexes[j]
        end_index = start_indexes[j] + int(np.ceil(temporal_n/ciclos))

        resultados[start_index:end_index] = resultados_temporales[::ciclos]
    
    return resultados

def conectar_db(nombre_db="resultados.db"):
    conn = sqlite3.connect(nombre_db)
    return conn

def verificar_tabla_existe(conn, nombre_tabla):
    query = '''
        SELECT name FROM sqlite_master WHERE type='table' AND name=?
    '''
    cursor = conn.cursor()
    cursor.execute(query, (nombre_tabla,))
    resultado = cursor.fetchone()
    return resultado is not None

def cargar_resultados(conn, inicio_tiempo, t_final):
    if not verificar_tabla_existe(conn, 'resultados'):
        print("La tabla 'resultados' no existe en la base de datos.")
        return None
    
    # Cargar datos desde la base de datos usando pandas
    query = '''
        SELECT * FROM resultados ORDER BY "index" ASC
    '''
    resultados_df = pd.read_sql_query(query, conn)
    
    # Convertir los resultados a un array de numpy
    if not resultados_df.empty:
        return resultados_df.drop(columns=['index']).to_numpy(dtype=np.float64)
    else:
        return None

def guardar_resultados(conn, resultados):
    # Convertir los resultados a un DataFrame
    columnas = ['v1_x', 'v1_y', 'v2_x', 'v2_y', 'v3_x', 'v3_y',
                'r1_x', 'r1_y', 'r2_x', 'r2_y', 'r3_x', 'r3_y']
    resultados_df = pd.DataFrame(resultados, columns=columnas)
    
    # Guardar resultados en la base de datos usando pandas
    resultados_df.to_sql('resultados', conn, if_exists='append', index=True)

def calcular_orbitas(t_inicio, t_final, n, r_inicial, v_inicial, m, limite_n, archivo=None):
    if archivo is not None:
        print(f"Buscando en el archivo: {archivo}")
        conn = conectar_db(archivo)
        resultados = cargar_resultados(conn, t_inicio, t_final)
        conn.close()
        if resultados is not None:
            return resultados
        else:
            print("No hay resultados en el archivo. Empezando desde cero...")

    resultados = calcular_ciclos(n, limite_n, t_inicio, t_final, v_inicial, r_inicial, m)

    if archivo is not None:
        conn = conectar_db(archivo)
        guardar_resultados(conn, resultados)
        conn.close()

    return resultados

import time

import os

# Verificar si el archivo existe
file_path = 'resultados_finales.csv'
if not os.path.exists(file_path):
    print(f"El archivo {file_path} no se encontró. Verifica la ruta.")
else:
    data_df = pd.read_csv(file_path)

# Leer los datos desde un archivo CSV
# Cargar los datos desde el archivo CSV
data_df = pd.read_csv('resultados_finales.csv')

oscila_data = data_df[data_df['oscila'] == 1]
random_data_file_path = 'datos_aleatorios.csv'
sample_data_df = pd.read_csv(random_data_file_path)
sample_data = sample_data_df[['pos_x', 'pos_y']]

# Establecer r_inicial y v_inicial
for i in range(sample_data.shape[0]):
    pos_x = sample_data_df.iloc[i]['pos_x']
    pos_y = sample_data_df.iloc[i]['pos_y']
    print(f'Posición {i}: pos_x = {pos_x}, pos_y = {pos_y}')
    r_inicial = np.array([
        [-0.5, 0.0], 
        [0.5, 0.0], 
        [pos_x, pos_y]
    ], dtype=np.float64)
    
    v_inicial = np.zeros((3, 2), dtype=np.float64)
    m = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    t_inicio = np.float64(0.0)
    t_final = 6.0  # Periodo fijo
    n = int((t_final - t_inicio) * 1.0e6)
    h = (t_final - t_inicio) / n
    limite_n = int(n * 0.001) if int(n * 0.001) < 1.0e5 else int(1.0e5)
    filename = f"resultados_{t_inicio:.2f}_{t_final:.2f}_{r_inicial[2][0]:.10f}_{r_inicial[2][1]:.10f}.db"

    start_time = time.time()
    try:
        resultados = calcular_orbitas(t_inicio, t_final, n, r_inicial, v_inicial, m, limite_n, archivo=filename)
    except Exception as e:
        print(f"Error al calcular órbitas: {e}")
    print(f"Tiempo de ejecución: {time.time() - start_time}s | Progreso: {i + 1}/{sample_data.shape[0]}")

    frame_indices = np.arange(0, resultados.shape[0], int(resultados.shape[0]*0.001))
    frame_data = resultados[frame_indices]

    # Animación
    fig, axis = plt.subplots()
    x_min, x_max = np.min(frame_data[:, 6:12:2]), np.max(frame_data[:, 6:12:2])
    y_min, y_max = np.min(frame_data[:, 7:13:2]), np.max(frame_data[:, 7:13:2])
    axis.set_xlim(x_min-0.1, x_max+0.1)
    axis.set_ylim(y_min-0.1, y_max+0.1)

    animated_plot_1, = axis.plot([], [], color="red")
    animated_mass_1, = axis.plot([], [], "o", markersize=3, color="red")
    animated_plot_2, = axis.plot([], [], color="blue")
    animated_mass_2, = axis.plot([], [], "o", markersize=3, color="blue")
    animated_plot_3, = axis.plot([], [], color="green")
    animated_mass_3, = axis.plot([], [], "o", markersize=3, color="green")

    def update_data(frame):

        if frame < len(frame_data):
            # Actualiza los plots de línea
            animated_plot_1.set_data(frame_data[:frame + 1, 6], frame_data[:frame + 1, 7])
            animated_plot_2.set_data(frame_data[:frame + 1, 8], frame_data[:frame + 1, 9])
            animated_plot_3.set_data(frame_data[:frame + 1, 10], frame_data[:frame + 1, 11])

            # Actualiza las posiciones de las masas, asegurando que son listas
            animated_mass_1.set_data([frame_data[frame, 6]], [frame_data[frame, 7]])
            animated_mass_2.set_data([frame_data[frame, 8]], [frame_data[frame, 9]])
            animated_mass_3.set_data([frame_data[frame, 10]], [frame_data[frame, 11]])

        return animated_plot_1, animated_plot_2, animated_plot_3, animated_mass_1, animated_mass_2, animated_mass_3

    animation = FuncAnimation(fig=fig, func=update_data, frames=len(frame_data), interval=1, repeat=False)
    # animation.save(f'{filename}.mp4', writer='ffmpeg', fps=30)

    plt.show()

