"""
Actividad: Aprendizaje no supervisado aplicado a rutas.

Sistema inteligente de rutas para analizar patrones de viaje
mediante el algoritmo K-Means.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler



# --------------------------------------------------------
# 1. BASE DE CONOCIMIENTO
# --------------------------------------------------------

base_conocimiento = {
    "Parque": {"zona": "comercial"},
    "Centro": {"zona": "comercial"},
    "Estacion": {"zona": "transporte"},
    "Barrio": {"zona": "residencial"},
    "Zona Industrial": {"zona": "industrial"}
}

rutas = {
    ("Parque", "Barrio"):
        ["Parque", "Centro", "Barrio"],

    ("Parque", "Zona Industrial"):
        ["Parque", "Centro", "Estacion", "Zona Industrial"],

    ("Centro", "Barrio"):
        ["Centro", "Estacion", "Barrio"],

    ("Centro", "Zona Industrial"):
        ["Centro", "Estacion", "Zona Industrial"]
}

METRO_INICIO = 4
METRO_FIN = 23


# --------------------------------------------------------
# 2. DATOS DE ENTRENAMIENTO SIMULADOS
# --------------------------------------------------------

# Cada registro representa un viaje.
# No incluimos etiquetas de riesgo.
# El algoritmo descubre los grupos por similitud.

datos_viajes = [
    # hora, lluvia, caminar_antes, caminar_despues,
    # metro, zona_industrial
    [8,  0, 3,  4, 1, 0],
    [9,  0, 5,  3, 1, 0],
    [10, 0, 2,  5, 1, 0],
    [14, 0, 4,  4, 1, 0],
    [16, 0, 6,  3, 1, 0],
    [11, 1, 3,  5, 1, 0],

    [18, 1, 8,  9, 1, 0],
    [19, 1, 12, 8, 1, 0],
    [20, 0, 9,  12, 1, 1],
    [21, 1, 11, 11, 1, 1],
    [22, 0, 8,  13, 1, 1],
    [19, 1, 10, 7, 1, 0],

    [2,  0, 3,  4, 1, 0],
    [3,  1, 5,  6, 1, 0],
    [1,  1, 12, 11, 1, 1],
    [23, 0, 4,  3, 1, 1],
    [20, 1, 2,  3, 0, 0],
    [13, 0, 3,  4, 0, 0]
]

columnas = [
    "hora",
    "lluvia",
    "caminar_antes",
    "caminar_despues",
    "metro",
    "zona_industrial"
]

df = pd.DataFrame(datos_viajes, columns=columnas)


# --------------------------------------------------------
# 3. PREPARACIÓN DE LOS DATOS
# --------------------------------------------------------

# K-Means utiliza distancias numéricas.
# Estandarizamos las variables para que tengan
# una escala comparable.

escalador = StandardScaler()

datos_escalados = escalador.fit_transform(df)


# --------------------------------------------------------
# 4. ENTRENAMIENTO NO SUPERVISADO
# --------------------------------------------------------

# K-Means busca tres grupos de viajes similares.
# No recibe categorías de riesgo como respuesta correcta.

modelo = KMeans(
    n_clusters=3,
    random_state=42,
    n_init=10
)

df["grupo"] = modelo.fit_predict(datos_escalados)


# --------------------------------------------------------
# 5. INTERPRETACIÓN DE LOS GRUPOS
# --------------------------------------------------------

# Calculamos características promedio de cada grupo.
# Esta interpretación se hace DESPUÉS del entrenamiento.

perfiles = df.groupby("grupo")[columnas].mean()

# Indicadores auxiliares para interpretar los grupos.
perfiles["madrugada"] = (
    (perfiles["hora"] < 5).astype(int)
)

perfiles["metro_fuera_horario"] = (
    (perfiles["metro"] > 0.5)
    & (
        (perfiles["hora"] < METRO_INICIO)
        | (perfiles["hora"] > METRO_FIN)
    )
).astype(int)

# Puntaje orientativo para ordenar los grupos.
# No es una probabilidad ni una predicción de delitos.

perfiles["puntaje"] = (
    perfiles["madrugada"] * 3
    + perfiles["lluvia"] * 1
    + perfiles["caminar_antes"] / 10
    + perfiles["caminar_despues"] / 10
    + perfiles["zona_industrial"] * 2
    + perfiles["metro_fuera_horario"] * 2
)

grupos_ordenados = perfiles["puntaje"].sort_values().index

clasificacion = {}

for posicion, grupo in enumerate(grupos_ordenados):
    if posicion == 0:
        clasificacion[grupo] = "PATRÓN FAVORABLE"
    elif posicion == 1:
        clasificacion[grupo] = "PATRÓN INTERMEDIO"
    else:
        clasificacion[grupo] = "PATRÓN EXIGENTE"


print("\n========== GRUPOS APRENDIDOS ==========")

for grupo in grupos_ordenados:
    print(f"\nGrupo {grupo}: {clasificacion[grupo]}")
    print(perfiles.loc[grupo, columnas].round(2))


# --------------------------------------------------------
# 6. CONSULTA DE UNA NUEVA RUTA
# --------------------------------------------------------

print("\n========== RUTAS DISPONIBLES ==========")

for origen_ruta, destino_ruta in rutas:
    print(f"- {origen_ruta} -> {destino_ruta}")

print("\n========== SELECCIÓN DE RUTA ==========")

origen = input("Punto de salida: ").strip()
destino = input("Punto de destino: ").strip()

clave = (origen, destino)

if clave not in rutas:
    print("\nNo existe esa ruta en el sistema.")
    print("Verifica el origen y el destino.")
    raise SystemExit

ruta = rutas[clave]

print("\n========== INFORMACIÓN DEL VIAJE ==========")

hora = int(input("Hora de salida (0-23): "))

clima = input(
    "Clima (soleado, nublado, lluvia): "
).strip().lower()

medio = input(
    "Transporte (metro, bus, indrive): "
).strip().lower()

caminar_antes = int(
    input("Minutos caminando antes del transporte: ")
)

caminar_despues = int(
    input("Minutos caminando después del transporte: ")
)

if not 0 <= hora <= 23:
    print("La hora debe estar entre 0 y 23.")
    raise SystemExit

if min(caminar_antes, caminar_despues) < 0:
    print("Los minutos caminando no pueden ser negativos.")
    raise SystemExit

if clima not in ["soleado", "nublado", "lluvia"]:
    print("El clima ingresado no es válido.")
    raise SystemExit

if medio not in ["metro", "bus", "indrive"]:
    print("El transporte ingresado no es válido.")
    raise SystemExit


# --------------------------------------------------------
# 7. TRANSFORMACIÓN DE LA NUEVA CONSULTA
# --------------------------------------------------------

lluvia = 1 if clima == "lluvia" else 0
usa_metro = 1 if medio == "metro" else 0

# Se identifica si la ruta pasa por una zona industrial.
zona_industrial = int(
    any(
        base_conocimiento[punto]["zona"] == "industrial"
        for punto in ruta
    )
)

nuevo_viaje = pd.DataFrame(
    [[
        hora,
        lluvia,
        caminar_antes,
        caminar_despues,
        usa_metro,
        zona_industrial
    ]],
    columns=columnas
)

nuevo_escalado = escalador.transform(nuevo_viaje)

# El modelo asigna el viaje al grupo más similar.
grupo_asignado = int(modelo.predict(nuevo_escalado)[0])

patron = clasificacion[grupo_asignado]


# --------------------------------------------------------
# 8. RESULTADOS
# --------------------------------------------------------

print("\n========== RESULTADO DEL ANÁLISIS ==========")

print("Ruta:", " -> ".join(ruta))
print("Transporte:", medio)
print("Grupo identificado:", grupo_asignado)
print("Patrón encontrado:", patron)

print("\nCaracterísticas del viaje:")
print("- Hora:", hora)
print("- Clima:", clima)
print("- Caminata antes:", caminar_antes, "minutos")
print("- Caminata después:", caminar_despues, "minutos")

print("\nInterpretación del grupo:")

if patron == "PATRÓN FAVORABLE":
    print(
        "El viaje se parece a otros recorridos con "
        "condiciones relativamente favorables."
    )
elif patron == "PATRÓN INTERMEDIO":
    print(
        "El viaje comparte características con recorridos "
        "que presentan algunas condiciones exigentes."
    )
else:
    print(
        "El viaje se parece a recorridos con varias "
        "condiciones que requieren atención."
    )

print(
    "\nNota: el grupo representa similitud entre viajes, "
    "no una medición real de seguridad."
)
# --------------------------------------------------------
# 9. REPRESENTACIÓN GRÁFICA DE LOS RESULTADOS
# --------------------------------------------------------

# Calculamos el tiempo total de caminata de cada viaje.
# Este valor se utiliza únicamente para representar los
# registros de forma sencilla en la primera gráfica.
df["caminata_total"] = (
    df["caminar_antes"] + df["caminar_despues"]
)


# --------------------------------------------------------
# GRÁFICA 1. GRUPOS IDENTIFICADOS POR K-MEANS
# --------------------------------------------------------

plt.figure(figsize=(10, 6))

# Recorremos los grupos descubiertos por K-Means.
for grupo in sorted(df["grupo"].unique()):

    # Seleccionamos únicamente los registros del grupo actual.
    datos_grupo = df[df["grupo"] == grupo]

    # Representamos la hora y la caminata total.
    plt.scatter(
        datos_grupo["hora"],
        datos_grupo["caminata_total"],
        s=90,
        label=f"Grupo {grupo}: {clasificacion[grupo]}"
    )

# Calculamos la caminata total de la consulta realizada.
caminata_total_nueva = caminar_antes + caminar_despues

# Marcamos la consulta del usuario con una estrella negra.
plt.scatter(
    hora,
    caminata_total_nueva,
    color="black",
    marker="*",
    s=300,
    label="Viaje consultado"
)

# Agregamos los elementos descriptivos de la gráfica.
plt.title("Grupos de viajes identificados mediante K-Means")
plt.xlabel("Hora de salida")
plt.ylabel("Minutos totales de caminata")
plt.xticks(range(0, 24))
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

# Mostramos la primera gráfica.
plt.show()


# --------------------------------------------------------
# GRÁFICA 2. CANTIDAD DE VIAJES POR GRUPO
# --------------------------------------------------------

# Contamos cuántos registros pertenecen a cada grupo.
cantidad_por_grupo = (
    df["grupo"]
    .value_counts()
    .sort_index()
)

# Creamos etiquetas que incluyen la interpretación del grupo.
etiquetas = [
    f"Grupo {grupo}\n{clasificacion[grupo]}"
    for grupo in cantidad_por_grupo.index
]

plt.figure(figsize=(10, 6))

# Creamos la gráfica de barras.
barras = plt.bar(
    etiquetas,
    cantidad_por_grupo.values,
    color=["#4CAF50", "#FFC107", "#F44336"]
)

# Agregamos la cantidad sobre cada barra.
for barra in barras:
    altura = barra.get_height()

    plt.text(
        barra.get_x() + barra.get_width() / 2,
        altura + 0.1,
        str(int(altura)),
        ha="center"
    )

# Agregamos los elementos descriptivos.
plt.title("Cantidad de viajes asignados a cada grupo")
plt.xlabel("Grupos identificados")
plt.ylabel("Cantidad de viajes")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

# Mostramos la segunda gráfica.
plt.show()