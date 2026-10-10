"""
Actividad 4 - Métodos de aprendizaje no supervisado.

Sistema inteligente de rutas para las líneas A y B del Metro de Medellín.

El programa combina:
- Una base de conocimiento con estaciones y conexiones.
- Reglas lógicas para validar los recorridos.
- El algoritmo A* para encontrar la ruta con menos tramos.
- El algoritmo K-Means para descubrir patrones entre viajes similares.
- Gráficas para visualizar los grupos encontrados.

Los datos utilizados para entrenar K-Means son simulados y tienen fines
exclusivamente académicos. Los grupos no representan  información oficial del Metro de Medellín.
"""

# heapq permite manejar la cola de prioridad utilizada por A*.
import heapq

# pandas permite organizar los datos simulados en una tabla.
import pandas as pd

# matplotlib permite crear las gráficas de los grupos obtenidos.
import matplotlib.pyplot as plt

# KMeans se utiliza para descubrir grupos de viajes similares.
from sklearn.cluster import KMeans

# StandardScaler coloca las variables en escalas comparables.
from sklearn.preprocessing import StandardScaler


# ============================================================
# 1. DATOS SIMULADOS DE VIAJES
# ============================================================

# Cada registro representa un viaje realizado anteriormente.
# El conjunto no contiene una etiqueta o respuesta correcta.
# K-Means deberá descubrir los grupos usando solamente estas variables:
#
# tramos: cantidad de desplazamientos entre estaciones.
# transferencias: cantidad de cambios entre las líneas A y B.
# hora_pico: 1 si el viaje ocurre en hora pico y 0 en caso contrario.

datos_viajes = [
    # Viajes de pocos tramos.
    [1, 0, 0],
    [2, 0, 0],
    [2, 0, 1],
    [3, 0, 0],
    [3, 0, 1],
    [4, 0, 0],
    [4, 0, 1],
    [5, 0, 0],
    [5, 0, 1],
    [6, 0, 0],

    # Viajes con una cantidad intermedia de tramos.
    [6, 1, 0],
    [7, 0, 1],
    [7, 1, 0],
    [8, 0, 0],
    [8, 1, 1],
    [9, 0, 1],
    [9, 1, 0],
    [10, 1, 1],
    [11, 0, 0],
    [11, 1, 1],

    # Viajes con una cantidad alta de tramos.
    [12, 0, 1],
    [12, 1, 0],
    [13, 1, 1],
    [14, 0, 0],
    [14, 1, 1],
    [15, 1, 0],
    [16, 1, 1],
    [17, 0, 1],
    [18, 1, 0],
    [19, 1, 1],
    [20, 1, 0],
    [21, 1, 1],
]

# Se asigna un nombre comprensible a cada columna del conjunto de datos.
columnas = ["tramos", "transferencias", "hora_pico"]

# Se convierte la lista de viajes en una tabla de pandas.
df = pd.DataFrame(datos_viajes, columns=columnas)


# ============================================================
# 2. PREPARACIÓN DE LOS DATOS
# ============================================================

""" K-Means compara los registros mediante distancias numéricas. La variable tramos puede llegar hasta 21, mientras que las otras variables solo contienen 0 o 1. Por eso se estandarizan los datos para evitar que una variable domine la agrupación únicamente por tener valores mayores."""

escalador = StandardScaler()

# El escalador aprende la escala de las variables y transforma los datos.
datos_escalados = escalador.fit_transform(df[columnas])


# ============================================================
# 3. ENTRENAMIENTO NO SUPERVISADO CON K-MEANS
# ============================================================

# Se crean tres grupos para obtener patrones cortos, intermedios y extensos.
# random_state permite obtener el mismo resultado cada vez que se ejecuta.
# n_init indica cuántas configuraciones iniciales prueba el algoritmo.

modelo = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10,
)

# fit_predict entrena el modelo y asigna un grupo a cada viaje simulado.
df["grupo"] = modelo.fit_predict(datos_escalados)


# ============================================================
# 4. INTERPRETACIÓN DE LOS GRUPOS
# ============================================================

"""K-Means devuelve números de grupo, pero no les asigna significados. Por eso, después del entrenamiento, se calculan los valores promedio de cada grupo para interpretar las características descubiertas."""

perfiles = df.groupby("grupo")[columnas].mean()

""" Este puntaje se utiliza solamente para ordenar e interpretar los grupos. No participa en el entrenamiento de K-Means.
Los tramos tienen el peso principal, mientras que una transferencia y la hora pico aumentan ligeramente la exigencia del recorrido."""

perfiles["puntaje"] = (
    perfiles["tramos"]
    + perfiles["transferencias"] * 3
    + perfiles["hora_pico"] * 2
)

# Se ordenan los grupos desde el puntaje promedio menor hasta el mayor.
grupos_ordenados = perfiles["puntaje"].sort_values().index.tolist()

# Se crea un diccionario para guardar el significado asignado a cada grupo.
clasificacion = {}

# Los nombres se asignan después de observar las características promedio.
for posicion, grupo in enumerate(grupos_ordenados):
    if posicion == 0:
        clasificacion[grupo] = "PATRÓN DE RECORRIDO CORTO"
    elif posicion == 1:
        clasificacion[grupo] = "PATRÓN DE RECORRIDO INTERMEDIO"
    else:
        clasificacion[grupo] = "PATRÓN DE RECORRIDO EXTENSO"


# ============================================================
# 5. BASE DE CONOCIMIENTO DEL METRO
# ============================================================

# San Antonio es el punto de transferencia representado entre A y B.
ESTACION_TRANSFERENCIA = "San Antonio"

# Estaciones de la Línea A organizadas de sur a norte.
LINEA_A = [
    "La Estrella",
    "Sabaneta",
    "Itagüí",
    "Envigado",
    "Ayurá",
    "Aguacatala",
    "Poblado",
    "Industriales",
    "Exposiciones",
    "Alpujarra",
    "San Antonio",
    "Parque Berrío",
    "Prado",
    "Hospital",
    "Universidad",
    "Caribe",
    "Tricentenario",
    "Acevedo",
    "Madera",
    "Bello",
    "Niquía",
]

# Estaciones de la Línea B organizadas desde San Antonio hasta San Javier.
LINEA_B = [
    "San Antonio",
    "Cisneros",
    "Suramericana",
    "Estadio",
    "Floresta",
    "Santa Lucía",
    "San Javier",
]

# Se agrupan las dos líneas para poder recorrerlas con el mismo código.
LINEAS = {
    "Línea A": LINEA_A,
    "Línea B": LINEA_B,
}

# Inicialmente no se registra ningún tramo cerrado.
TRAMOS_CERRADOS = set()

# Horas pico definidas únicamente como criterio académico del ejercicio.
HORAS_PICO = {5, 6, 7, 8, 16, 17, 18, 19}


# ============================================================
# 6. REGLAS LÓGICAS
# ============================================================

def regla_estaciones_consecutivas(estacion_1, estacion_2, estaciones_linea):
    """Comprueba si dos estaciones son consecutivas en una línea."""

    posicion_1 = estaciones_linea.index(estacion_1)
    posicion_2 = estaciones_linea.index(estacion_2)

    return abs(posicion_1 - posicion_2) == 1


def regla_tramo_disponible(estacion_1, estacion_2):
    """Permite utilizar un tramo cuando no está registrado como cerrado."""

    # frozenset permite reconocer el mismo tramo en ambos sentidos.
    tramo = frozenset((estacion_1, estacion_2))

    return tramo not in TRAMOS_CERRADOS


def regla_transferencia_permitida(estacion):
    """Permite cambiar entre las líneas A y B solamente en San Antonio."""

    return estacion == ESTACION_TRANSFERENCIA


# ============================================================
# 7. CONSTRUCCIÓN DEL GRAFO
# ============================================================

def construir_grafo():
    """Convierte las líneas y sus estaciones en un grafo de conexiones."""

    # El diccionario guardará las estaciones vecinas de cada estación.
    grafo = {}

    # Se procesa cada línea de manera independiente.
    for nombre_linea, estaciones in LINEAS.items():
        # Cada estación se conecta con la que aparece inmediatamente después.
        for indice in range(len(estaciones) - 1):
            estacion_actual = estaciones[indice]
            estacion_siguiente = estaciones[indice + 1]

            # Se aplica la regla que exige que sean estaciones consecutivas.
            if regla_estaciones_consecutivas(
                estacion_actual,
                estacion_siguiente,
                estaciones,
            ):
                # Se crean las listas de conexiones si todavía no existen.
                grafo.setdefault(estacion_actual, [])
                grafo.setdefault(estacion_siguiente, [])

                # Cada tramo tiene costo 1 y puede recorrerse en ambos sentidos.
                grafo[estacion_actual].append(
                    (estacion_siguiente, nombre_linea, 1)
                )
                grafo[estacion_siguiente].append(
                    (estacion_actual, nombre_linea, 1)
                )

    return grafo


# Se construye una vez el grafo utilizado por el algoritmo A*.
GRAFO = construir_grafo()

# Este índice permite conocer las líneas y posiciones de cada estación.
POSICIONES = {}

for nombre_linea, estaciones in LINEAS.items():
    for posicion, estacion in enumerate(estaciones):
        POSICIONES.setdefault(estacion, []).append(
            (nombre_linea, posicion)
        )

# Se almacena la posición de San Antonio en cada línea.
POSICION_TRANSFERENCIA = {
    "Línea A": LINEA_A.index(ESTACION_TRANSFERENCIA),
    "Línea B": LINEA_B.index(ESTACION_TRANSFERENCIA),
}


# ============================================================
# 8. HEURÍSTICA Y ALGORITMO A*
# ============================================================

def heuristica(estacion_actual, destino):
    """Estima los tramos restantes desde una estación hasta el destino."""

    estimaciones = []

    # Se revisan las líneas a las que pertenecen las dos estaciones.
    for linea_actual, posicion_actual in POSICIONES[estacion_actual]:
        for linea_destino, posicion_destino in POSICIONES[destino]:
            # Si están en la misma línea, se calcula la diferencia de posición.
            if linea_actual == linea_destino:
                estimacion = abs(posicion_actual - posicion_destino)
            else:
                # Si están en líneas diferentes, se estima el paso por
                # la estación de transferencia San Antonio.
                hasta_transferencia = abs(
                    posicion_actual
                    - POSICION_TRANSFERENCIA[linea_actual]
                )
                desde_transferencia = abs(
                    posicion_destino
                    - POSICION_TRANSFERENCIA[linea_destino]
                )
                estimacion = (
                    hasta_transferencia + desde_transferencia
                )

            estimaciones.append(estimacion)

    return min(estimaciones)


def buscar_mejor_ruta(inicio, destino):
    """Ejecuta A* y devuelve la ruta permitida con menos tramos."""

    # Cada alternativa contiene prioridad, costo, estación, ruta y líneas.
    frontera = []

    heapq.heappush(
        frontera,
        (heuristica(inicio, destino), 0, inicio, [inicio], []),
    )

    # Se guarda el menor costo conocido para llegar a cada estación.
    mejor_costo = {inicio: 0}

    while frontera:
        # Se extrae la alternativa con el menor valor estimado de f(n).
        _, costo_actual, estacion_actual, ruta, lineas_usadas = (
            heapq.heappop(frontera)
        )

        # Cuando se alcanza el destino, se devuelve la solución.
        if estacion_actual == destino:
            return ruta, lineas_usadas, costo_actual

        # Se descarta una alternativa si ya se conoce otra más económica.
        if costo_actual > mejor_costo.get(
            estacion_actual,
            float("inf"),
        ):
            continue

        # Se revisan las estaciones vecinas que permite el grafo.
        for estacion_vecina, linea, costo_tramo in GRAFO[estacion_actual]:
            # La regla elimina de la búsqueda los tramos cerrados.
            if not regla_tramo_disponible(
                estacion_actual,
                estacion_vecina,
            ):
                continue

            nuevo_costo = costo_actual + costo_tramo

            # Se guarda la nueva alternativa si mejora el costo conocido.
            if nuevo_costo < mejor_costo.get(
                estacion_vecina,
                float("inf"),
            ):
                mejor_costo[estacion_vecina] = nuevo_costo

                prioridad = (
                    nuevo_costo
                    + heuristica(estacion_vecina, destino)
                )

                heapq.heappush(
                    frontera,
                    (
                        prioridad,
                        nuevo_costo,
                        estacion_vecina,
                        ruta + [estacion_vecina],
                        lineas_usadas + [linea],
                    ),
                )

    # Si se agotan las alternativas, no existe una ruta disponible.
    return None, None, None


# ============================================================
# 9. FUNCIONES DE APOYO
# ============================================================

def obtener_estaciones():
    """Crea una lista única de estaciones para el menú."""

    # dict.fromkeys elimina el San Antonio repetido y conserva el orden.
    return list(dict.fromkeys(LINEA_A + LINEA_B))


def seleccionar_estacion(mensaje, estaciones):
    """Muestra el menú y solicita una estación válida."""

    while True:
        print(f"\n{mensaje}")

        for numero, estacion in enumerate(estaciones, start=1):
            print(f"{numero:2}. {estacion}")

        respuesta = input("Escriba el número de la estación: ").strip()

        if respuesta.isdigit():
            opcion = int(respuesta)

            if 1 <= opcion <= len(estaciones):
                return estaciones[opcion - 1]

        print("Opción no válida. Intente nuevamente.")


def seleccionar_hora():
    """Solicita y valida una hora entre 0 y 23."""

    while True:
        respuesta = input("Hora de salida (0-23): ").strip()

        if respuesta.isdigit():
            hora = int(respuesta)

            if 0 <= hora <= 23:
                return hora

        print("Hora no válida. Escriba un número entre 0 y 23.")


def obtener_transferencias(ruta, lineas_usadas):
    """Obtiene las estaciones donde cambia la línea utilizada."""

    transferencias = []

    for indice in range(1, len(lineas_usadas)):
        if lineas_usadas[indice] != lineas_usadas[indice - 1]:
            estacion_cambio = ruta[indice]

            if regla_transferencia_permitida(estacion_cambio):
                transferencias.append(estacion_cambio)

    return transferencias


# ============================================================
# 10. GRÁFICAS DE LOS RESULTADOS
# ============================================================

def graficar_resultados(
    datos,
    nuevo_viaje,
    grupo_asignado,
    nombres_grupos,
):
    """Muestra dos gráficas para interpretar los grupos de viajes."""

    # Primera gráfica: distribución por tramos y transferencias.
    plt.figure(figsize=(10, 6))

    for grupo in sorted(datos["grupo"].unique()):
        datos_grupo = datos[datos["grupo"] == grupo]

        plt.scatter(
            datos_grupo["tramos"],
            datos_grupo["transferencias"],
            s=80,
            alpha=0.75,
            label=nombres_grupos[grupo],
        )

    # La estrella negra representa la consulta realizada por el usuario.
    plt.scatter(
        nuevo_viaje["tramos"].iloc[0],
        nuevo_viaje["transferencias"].iloc[0],
        color="black",
        marker="*",
        s=300,
        label="Recorrido consultado",
    )

    plt.title("Grupos de viajes descubiertos con K-Means")
    plt.xlabel("Cantidad de tramos")
    plt.ylabel("Cantidad de transferencias")
    plt.yticks([0, 1])
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Segunda gráfica: cantidad de viajes asignados a cada grupo.
    conteo = datos["grupo"].value_counts().sort_index()
    etiquetas = [nombres_grupos[grupo] for grupo in conteo.index]

    plt.figure(figsize=(10, 5))

    barras = plt.bar(
        etiquetas,
        conteo.values,
        color=["#4C78A8", "#F2CF5B", "#E45756"],
    )

    # Se escribe sobre cada barra la cantidad de registros del grupo.
    for barra, cantidad in zip(barras, conteo.values):
        plt.text(
            barra.get_x() + barra.get_width() / 2,
            cantidad + 0.15,
            str(cantidad),
            ha="center",
        )

    plt.title("Cantidad de viajes por grupo")
    plt.xlabel("Patrón identificado")
    plt.ylabel("Cantidad de viajes")
    plt.xticks(rotation=10)
    plt.tight_layout()
    plt.show()


# ============================================================
# 11. FUNCIÓN PRINCIPAL
# ============================================================

def ejecutar_sistema():
    """Coordina la selección, la búsqueda y el análisis no supervisado."""

    print("=" * 68)
    print("SISTEMA INTELIGENTE DE RUTAS - METRO DE MEDELLÍN")
    print("Búsqueda A* y aprendizaje no supervisado con K-Means")
    print("=" * 68)

    # Se muestran los perfiles aprendidos por el modelo.
    print("\n========== GRUPOS APRENDIDOS ==========")

    for grupo in grupos_ordenados:
        print(f"\nGrupo {grupo}: {clasificacion[grupo]}")
        print(perfiles.loc[grupo, columnas].round(2))

    # El usuario selecciona el inicio y el destino del recorrido.
    estaciones = obtener_estaciones()
    inicio = seleccionar_estacion("INICIO DEL RECORRIDO", estaciones)
    destino = seleccionar_estacion("DESTINO DEL RECORRIDO", estaciones)

    # Si ambas estaciones son iguales, no existe un viaje para analizar.
    if inicio == destino:
        print("\nEl inicio y el destino son iguales.")
        print("No se requiere recorrido ni análisis de agrupamiento.")
        return

    # Se solicita la hora para determinar el indicador académico de hora pico.
    hora_salida = seleccionar_hora()
    hora_pico = 1 if hora_salida in HORAS_PICO else 0

    # A* encuentra la ruta válida con la menor cantidad de tramos.
    ruta, lineas_usadas, total_tramos = buscar_mejor_ruta(
        inicio,
        destino,
    )

    if ruta is None:
        print("\nNo existe una ruta disponible con los tramos habilitados.")
        return

    # Se calculan las transferencias utilizadas por la ruta encontrada.
    estaciones_transferencia = obtener_transferencias(
        ruta,
        lineas_usadas,
    )
    cantidad_transferencias = len(estaciones_transferencia)

    # La nueva consulta se organiza con las mismas variables del entrenamiento.
    nuevo_viaje = pd.DataFrame(
        [[total_tramos, cantidad_transferencias, hora_pico]],
        columns=columnas,
    )

    # El nuevo registro se transforma con el escalador ya entrenado.
    nuevo_escalado = escalador.transform(nuevo_viaje)

    # K-Means asigna el viaje al grupo con el centroide más cercano.
    grupo_asignado = int(modelo.predict(nuevo_escalado)[0])
    patron_asignado = clasificacion[grupo_asignado]

    # Se presentan los resultados obtenidos por A* y K-Means.
    print("\n========== RESULTADO DEL ANÁLISIS ==========")
    print("Inicio del recorrido:", inicio)
    print("Destino del recorrido:", destino)
    print("Mejor ruta:", " -> ".join(ruta))
    print("Cantidad de tramos:", total_tramos)
    print("Cantidad de transferencias:", cantidad_transferencias)
    print("Hora de salida:", hora_salida)
    print("¿Hora pico?:", "Sí" if hora_pico == 1 else "No")
    print("Grupo identificado:", grupo_asignado)
    print("Patrón encontrado:", patron_asignado)

    if estaciones_transferencia:
        print(
            "Transferencia necesaria en:",
            ", ".join(estaciones_transferencia),
        )
    else:
        print("El recorrido no requiere transferencia entre líneas.")

    print("\nInterpretación:")

    if patron_asignado == "PATRÓN DE RECORRIDO CORTO":
        print(
            "El viaje se parece a los recorridos simulados de pocos "
            "tramos y menor complejidad de desplazamiento."
        )
    elif patron_asignado == "PATRÓN DE RECORRIDO INTERMEDIO":
        print(
            "El viaje comparte características con recorridos simulados "
            "de extensión intermedia."
        )
    else:
        print(
            "El viaje se parece a los recorridos simulados con mayor "
            "cantidad de tramos o condiciones adicionales."
        )

    print(
        "\nNota: el grupo representa similitud entre viajes simulados; "
        "no corresponde a una medición oficial del servicio."
    )

    # Finalmente se generan las dos gráficas exigidas en la actividad.
    graficar_resultados(
        df,
        nuevo_viaje,
        grupo_asignado,
        clasificacion,
    )


# Esta condición ejecuta el sistema al abrir directamente el archivo.
if __name__ == "__main__":
    ejecutar_sistema()
