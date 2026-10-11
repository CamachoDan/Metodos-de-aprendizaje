# ============================================================
# SISTEMA DE ANALISIS DE RUTAS UTILIZANDO REGRESION LINEAL
# ============================================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


def main():

    # --------------------------------------------------------
    # 1. DATASET: BASE DE DATOS HISTORICA SIMULADA
    # --------------------------------------------------------

    # Cada registro representa una observacion de una ruta.
    # Los datos son ficticios y se utilizan para fines educativos.

    datos = pd.DataFrame({

        "hora": [
            8, 9, 12, 14, 17,
            19, 20, 22, 23, 6,
            10, 15, 18, 21, 7,
            11, 16, 20, 22, 13,
            5, 14, 19, 21, 9,
            17, 23, 12, 18, 22
        ],

        "caminar_minutos": [
            5, 8, 5, 12, 15,
            8, 15, 20, 12, 10,
            4, 8, 10, 18, 6,
            14, 6, 10, 15, 7,
            18, 5, 12, 20, 3,
            10, 15, 6, 14, 18
        ],

        # 0 = no llueve, 1 = llueve
        "lluvia": [
            0, 0, 1, 0, 1,
            0, 1, 1, 0, 1,
            0, 1, 0, 1, 0,
            1, 0, 0, 1, 0,
            1, 0, 1, 1, 0,
            0, 1, 0, 1, 1
        ],

        # Cantidad de incidentes historicos registrados
        "incidentes": [
            0, 0, 0, 1, 1,
            2, 3, 4, 3, 1,
            0, 1, 2, 3, 0,
            1, 1, 2, 4, 0,
            2, 0, 3, 4, 0,
            1, 4, 0, 2, 3
        ],

        # Variable objetivo: riesgo registrado entre 0 y 100
        "riesgo": [
            10, 12, 18, 30, 48,
            52, 82, 95, 78, 32,
            8, 35, 50, 85, 10,
            38, 25, 58, 92, 12,
            45, 9, 65, 90, 7,
            35, 94, 15, 60, 88
        ]
    })

    print("\n========== DATASET HISTORICO ==========")
    print(datos.to_string(index=False))

    print("\nCantidad de registros:", len(datos))

    # --------------------------------------------------------
    # 2. SELECCION DE VARIABLES
    # --------------------------------------------------------

    # Variables de entrada utilizadas para realizar la prediccion.
    X = datos[
        ["hora", "caminar_minutos", "lluvia", "incidentes"]
    ]

    # Variable que el modelo aprende a estimar.
    y = datos["riesgo"]

    # --------------------------------------------------------
    # 3. DIVISION DE LOS DATOS
    # --------------------------------------------------------

    # Separamos los datos para entrenar y evaluar el modelo.
    X_entrenamiento, X_prueba, y_entrenamiento, y_prueba = (
        train_test_split(
            X,
            y,
            test_size=0.25,
            random_state=42
        )
    )

    # --------------------------------------------------------
    # 4. ENTRENAMIENTO DE LA REGRESION LINEAL
    # --------------------------------------------------------

    modelo = LinearRegression()

    # El modelo aprende la relacion entre las variables
    # de entrada y los niveles de riesgo del dataset.
    modelo.fit(X_entrenamiento, y_entrenamiento)

    print("\n========== MODELO ENTRENADO ==========")
    print("Modelo: Regresion lineal")

    print("\nCoeficientes del modelo:")

    for variable, coeficiente in zip(
        X.columns, modelo.coef_
    ):
        print(f"{variable}: {coeficiente:.4f}")

    print("Intercepto:", round(modelo.intercept_, 4))

    # --------------------------------------------------------
    # 5. EVALUACION DEL MODELO
    # --------------------------------------------------------

    predicciones_prueba = modelo.predict(X_prueba)

    mae = mean_absolute_error(
        y_prueba,
        predicciones_prueba
    )

    r2 = r2_score(
        y_prueba,
        predicciones_prueba
    )

    print("\n========== EVALUACION DEL MODELO ==========")
    print(f"Error absoluto medio (MAE): {mae:.2f}")
    print(f"Coeficiente de determinacion R²: {r2:.2f}")

    print(
        "\nEl MAE representa el error promedio en puntos "
        "de riesgo."
    )

    print(
        "El R² indica que tan bien se ajusta el modelo "
        "a los datos de evaluacion."
    )

    # --------------------------------------------------------
    # 6. BASE DE CONOCIMIENTO DE LAS RUTAS
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

    # --------------------------------------------------------
    # 7. SELECCION DE LA RUTA
    # --------------------------------------------------------

    print("\n========== RUTAS DISPONIBLES ==========")

    for origen_ruta, destino_ruta in rutas:
        print(f"- {origen_ruta} -> {destino_ruta}")

    print("\n========== SELECCION DE RUTA ==========")

    origen = input("Punto de salida: ").strip()
    destino = input("Punto de destino: ").strip()

    clave = (origen, destino)

    if clave not in rutas:

        print("\nNo existe esa ruta en el sistema.")

        print("\nRutas disponibles:")

        for origen_ruta, destino_ruta in rutas:
            print(f"- {origen_ruta} -> {destino_ruta}")

        return

    ruta = rutas[clave]

    # --------------------------------------------------------
    # 8. INGRESO DE LAS CONDICIONES DEL RECORRIDO
    # --------------------------------------------------------

    print("\n========== INFORMACION DEL VIAJE ==========")

    try:

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
            input("Minutos caminando despues del transporte: ")
        )

        incidentes = int(
            input("Incidentes historicos registrados en la zona: ")
        )

    except ValueError:

        print("\nError: ingresa valores numericos validos.")
        return

    # Validacion de los datos ingresados.
    if not 0 <= hora <= 23:
        print("La hora debe estar entre 0 y 23.")
        return

    if caminar_antes < 0 or caminar_despues < 0:
        print("Los minutos caminando no pueden ser negativos.")
        return

    if incidentes < 0:
        print("Los incidentes no pueden ser negativos.")
        return

    if clima not in ["soleado", "nublado", "lluvia"]:
        print("El clima ingresado no es valido.")
        return

    if medio not in ["metro", "bus", "indrive"]:
        print("El medio de transporte no es valido.")
        return

    # Convertimos el clima a una variable numerica.
    lluvia = 1 if clima == "lluvia" else 0

    # Calculamos el tiempo total caminando.
    caminar_total = caminar_antes + caminar_despues

    # --------------------------------------------------------
    # 9. PREDICCION DEL RIESGO CON REGRESION LINEAL
    # --------------------------------------------------------

    nueva_ruta = pd.DataFrame([{

        "hora": hora,
        "caminar_minutos": caminar_total,
        "lluvia": lluvia,
        "incidentes": incidentes

    }])

    # El modelo estima el riesgo utilizando las variables
    # ingresadas y la relacion aprendida del dataset.
    prediccion = modelo.predict(nueva_ruta)[0]

    # Limitamos el resultado a la escala de 0 a 100.
    riesgo = max(0, min(100, prediccion))

    # --------------------------------------------------------
    # 10. REGLAS COMPLEMENTARIAS DE SEGURIDAD
    # --------------------------------------------------------

    METRO_INICIO = 4
    METRO_FIN = 23

    BUS_INICIO = 5
    BUS_FIN = 22

    LIMITE_CAMINAR = 10

    condiciones_riesgo = []
    recomendaciones = []

    # Condicion de madrugada.
    if 0 <= hora < 5:

        condiciones_riesgo.append(
            "La salida se realiza durante la madrugada."
        )

        recomendaciones.append(
            "Evaluar la posibilidad de cambiar el horario."
        )

    # Zona industrial durante la noche.
    if hora >= 19 or hora < 5:

        for punto in ruta:

            zona = base_conocimiento[punto]["zona"]

            if zona == "industrial":

                condiciones_riesgo.append(
                    "La ruta pasa por una zona industrial de noche."
                )

                recomendaciones.append(
                    "Evaluar una ruta alternativa."
                )

                break

    # Condicion de lluvia.
    if lluvia and caminar_total > 0:

        condiciones_riesgo.append(
            "La lluvia puede dificultar el desplazamiento."
        )

        recomendaciones.append(
            "Reducir el tiempo caminando cuando sea posible."
        )

    # Disponibilidad del transporte.
    if medio == "metro":

        if METRO_INICIO <= hora <= METRO_FIN:

            print("\nEl Metro esta dentro del horario configurado.")

        else:

            condiciones_riesgo.append(
                "El Metro esta fuera del horario configurado."
            )

            recomendaciones.append(
                "Consultar otro medio de transporte."
            )

    elif medio == "bus":

        if BUS_INICIO <= hora <= BUS_FIN:

            print("\nEl bus esta dentro del horario configurado.")

        else:

            condiciones_riesgo.append(
                "El bus esta fuera del horario configurado."
            )

            recomendaciones.append(
                "Consultar otro medio de transporte."
            )

    # Tiempo caminando antes del transporte.
    if caminar_antes > LIMITE_CAMINAR:

        condiciones_riesgo.append(
            f"Se caminan {caminar_antes} minutos antes del transporte."
        )

        recomendaciones.append(
            "Buscar un punto de transporte mas cercano."
        )

    # Tiempo caminando despues del transporte.
    if caminar_despues > LIMITE_CAMINAR:

        condiciones_riesgo.append(
            f"Se caminan {caminar_despues} minutos despues del transporte."
        )

        recomendaciones.append(
            "Buscar un transporte mas cercano al destino."
        )

    # Eliminamos elementos repetidos.
    condiciones_riesgo = list(dict.fromkeys(condiciones_riesgo))
    recomendaciones = list(dict.fromkeys(recomendaciones))

    # --------------------------------------------------------
    # 11. CLASIFICACION DEL RESULTADO
    # --------------------------------------------------------

    # El nivel se basa en la prediccion del modelo.
    if riesgo < 30:

        nivel = "RIESGO BAJO"

    elif riesgo < 60:

        nivel = "RIESGO MODERADO"

    else:

        nivel = "RIESGO ALTO"

    # --------------------------------------------------------
    # 12. MOSTRAR LOS RESULTADOS
    # --------------------------------------------------------

    print("\n========== RESULTADO DEL ANALISIS ==========")

    print("Ruta:", " -> ".join(ruta))
    print("Transporte:", medio)
    print("Hora de salida:", hora)
    print("Clima:", clima)
    print("Minutos caminando:", caminar_total)
    print("Incidentes historicos ingresados:", incidentes)

    print(f"\nRiesgo estimado: {riesgo:.2f}/100")
    print("Nivel:", nivel)

    print("\nCondiciones identificadas:")

    if condiciones_riesgo:

        for condicion in condiciones_riesgo:
            print("-", condicion)

    else:

        print("- No se identifican condiciones adicionales.")

    print("\nRecomendaciones:")

    if recomendaciones:

        for recomendacion in recomendaciones:
            print("-", recomendacion)

    else:

        print("- No se generan recomendaciones adicionales.")

    print(
        "\nNota: el resultado es una estimacion basada en datos "
        "simulados y no garantiza la seguridad de la ruta."
    )

    # --------------------------------------------------------
    # 13. GRAFICA DE REGRESION LINEAL CON LA ESTIMACION (X)
    # --------------------------------------------------------

    # Generamos diferentes escenarios de tiempo caminando,
    # manteniendo constantes las demas condiciones de la ruta.

    minutos = list(range(0, max(30, caminar_total + 5) + 1))

    escenarios = pd.DataFrame({
        "hora": [hora] * len(minutos),
        "caminar_minutos": minutos,
        "lluvia": [lluvia] * len(minutos),
        "incidentes": [incidentes] * len(minutos)
    })

    # Predicciones del modelo para cada tiempo caminando.
    riesgos_estimados = modelo.predict(escenarios)

    # Limitar los resultados al rango de 0 a 100.
    riesgos_estimados = [
        max(0, min(100, valor))
        for valor in riesgos_estimados
    ]

    # Crear la figura.
    plt.figure(figsize=(10, 6))

    # Mostrar los registros del dataset.
    plt.scatter(
        datos["caminar_minutos"],
        datos["riesgo"],
        color="blue",
        alpha=0.6,
        label="Datos del dataset"
    )

    # Dibujar la curva estimada por el modelo.
    plt.plot(
        minutos,
        riesgos_estimados,
        color="green",
        linewidth=2,
        label="Regresion lineal"
    )

    # Marcar la estimacion de la ruta con una X roja.
    plt.scatter(
        caminar_total,
        riesgo,
        color="red",
        marker="x",
        s=110,
        linewidths=1,
        zorder=5,
        label="Estimacion de tu ruta"
    )

    # Mostrar el valor estimado al lado de la X.
    plt.annotate(
        f"Riesgo: {riesgo:.2f}",
        xy=(caminar_total, riesgo),
        xytext=(10, 12),
        textcoords="offset points",
        fontsize=10,
        color="red",
        fontweight="light"
    )

    plt.xlabel("Minutos caminando")
    plt.ylabel("Riesgo (0-100)")

    plt.title(
        "Regresion lineal: estimacion del riesgo de la ruta"
    )

    plt.xlim(0, max(30, caminar_total + 5))
    plt.ylim(0, 100)

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()


    # --------------------------------------------------------
    # 14. GRAFICA DEL RIESGO SEGUN EL TIEMPO CAMINANDO
    # --------------------------------------------------------

    plt.figure(figsize=(10, 6))

    # Curva de riesgo estimado.
    plt.plot(
        minutos,
        riesgos_estimados,
        color="green",
        linewidth=2,
        label="Riesgo estimado"
    )

    # X que identifica el tiempo caminando y el riesgo
    # estimado para la ruta ingresada.
    plt.scatter(
        [caminar_total],
        [riesgo],
        color="red",
        marker="x",
        s=110,
        linewidths=1,
        zorder=5,
        label="Tu ruta"
    )

    plt.annotate(
        f"({caminar_total} min, {riesgo:.2f})",
        xy=(caminar_total, riesgo),
        xytext=(10, 12),
        textcoords="offset points",
        fontsize=10,
        color="red",
        fontweight="light"
    )

    plt.xlabel("Tiempo total caminando (minutos)")
    plt.ylabel("Riesgo estimado (0-100)")

    plt.title(
        "Riesgo estimado segun el tiempo caminando"
    )

    # Ajustar los ejes para mostrar la estimacion.
    plt.xlim(0, max(30, caminar_total + 5))
    plt.ylim(0, 100)

    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    # Mostrar ambas graficas.
    plt.show()

# ------------------------------------------------------------
# 15. EJECUTAR EL PROGRAMA
# ------------------------------------------------------------

if __name__ == "__main__":
    main()