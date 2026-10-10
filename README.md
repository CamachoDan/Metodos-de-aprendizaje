# Métodos de aprendizaje aplicados al análisis de rutas

## 1. Descripción

Este repositorio contiene los códigos desarrollados para aplicar métodos de aprendizaje supervisado y no supervisado a dos sistemas inteligentes de análisis de rutas.

Los sistemas base son:

1. **Sistema de análisis de condiciones y riesgo de una ruta**, que considera factores como la hora de salida, el clima, el tiempo de caminata, el medio de transporte, los incidentes históricos simulados y las zonas recorridas.

2. **Sistema inteligente de rutas del Metro de Medellín**, que representa estaciones de las líneas A y B, utiliza reglas lógicas para validar las conexiones y aplica el algoritmo A* para encontrar el recorrido con la menor cantidad de tramos.

A partir de estos sistemas se desarrollaron dos actividades:

- **Actividad 3:** aplicación de aprendizaje supervisado mediante regresión lineal.
- **Actividad 4:** aplicación de aprendizaje no supervisado mediante K-Means.

Los datasets utilizados son simulados, se encuentran incorporados directamente en los cuadernos de Google Colab y en los archivos `.py`, y se emplean únicamente con fines académicos.

---

## 2. Objetivo

Aplicar métodos de aprendizaje automático a sistemas de análisis de rutas, integrando modelos supervisados y no supervisados con reglas lógicas y algoritmos de búsqueda.

Los objetivos específicos son:

- Estimar valores numéricos mediante modelos de regresión lineal.
- Identificar grupos de viajes con características similares mediante K-Means.
- Integrar los modelos con los sistemas de rutas desarrollados previamente.
- Generar resultados textuales y representaciones gráficas.
- Analizar el comportamiento de los modelos mediante datos simulados.
- Aplicar conceptos de aprendizaje supervisado, aprendizaje no supervisado, reglas lógicas y búsqueda heurística.

---

## 3. Componentes del repositorio

El repositorio contiene cuatro desarrollos principales:

| Actividad | Sistema | Método aplicado |
|---|---|---|
| Actividad 3 | Análisis de condiciones y riesgo de una ruta | Regresión lineal |
| Actividad 3 | Rutas del Metro de Medellín con A* | Regresión lineal |
| Actividad 4 | Análisis de condiciones de una ruta | K-Means |
| Actividad 4 | Rutas del Metro de Medellín con A* | K-Means |

---

# Actividad 3 Métodos de aprendizaje supervisado

## 4. Aprendizaje supervisado

El aprendizaje supervisado utiliza datos que contienen variables de entrada y un resultado conocido. A partir de estos ejemplos, el modelo aprende una relación que posteriormente utiliza para generar predicciones sobre nuevas consultas.

En esta actividad se utiliza el algoritmo de **regresión lineal**, debido a que los resultados esperados son valores numéricos continuos:

- Un valor estimado de riesgo para el sistema de análisis de condiciones.
- Un tiempo estimado de recorrido para el sistema del Metro de Medellín.

---

## 5. Sistema de análisis de rutas mediante regresión lineal

### 5.1. Descripción

Este sistema utiliza regresión lineal para estimar un valor numérico de riesgo relacionado con las condiciones de un viaje.

El usuario selecciona una ruta e ingresa información como:

- Hora de salida.
- Estado del clima.
- Medio de transporte.
- Minutos caminando antes del transporte.
- Minutos caminando después del transporte.
- Cantidad de incidentes históricos simulados.

El modelo genera una estimación y el sistema la complementa con reglas lógicas relacionadas con:

- Recorridos realizados durante la madrugada.
- Paso por una zona industrial durante la noche.
- Lluvia y desplazamientos caminando.
- Disponibilidad del Metro o del bus según el horario configurado.
- Caminatas superiores al límite establecido.

Finalmente, el programa presenta:

- Ruta seleccionada.
- Medio de transporte.
- Valor estimado por la regresión lineal.
- Nivel calculado.
- Condiciones identificadas.
- Recomendaciones.
- Gráficas de los resultados.

### 5.2. Dataset incorporado

El dataset contiene 30 registros simulados.

Las variables de entrada son:

| Variable | Descripción |
|---|---|
| `hora` | Hora de salida del viaje |
| `caminar_minutos` | Minutos asociados con el desplazamiento caminando |
| `lluvia` | Indica si está lloviendo: 1 significa sí y 0 significa no |
| `incidentes` | Cantidad simulada de incidentes históricos |

La variable objetivo es:

| Variable | Descripción |
|---|---|
| `riesgo` | Valor numérico simulado entre 0 y 100 |

Los datos son ficticios y no corresponden a registros oficiales de seguridad, movilidad o transporte.

---

## 6. Sistema del Metro mediante A* y regresión lineal

### 6.1. Descripción

Este sistema representa estaciones y conexiones de las líneas A y B del Metro de Medellín.

Las estaciones corresponden a los nodos del grafo y las conexiones entre estaciones consecutivas corresponden a las aristas. Cada desplazamiento tiene un costo de un tramo y las conexiones pueden recorrerse en ambos sentidos.

El sistema conserva los siguientes elementos:

- Base de conocimiento de las estaciones.
- Reglas lógicas para validar las conexiones.
- Identificación de tramos disponibles.
- Transferencia entre las líneas A y B en San Antonio.
- Algoritmo A* para encontrar la ruta con la menor cantidad de tramos.
- Validación de las opciones ingresadas.
- Identificación de la hora pico mediante una regla académica.

Después de encontrar la ruta, el modelo de regresión lineal estima el tiempo del recorrido.

El resultado incluye:

- Estación de inicio.
- Estación de destino.
- Mejor ruta encontrada.
- Cantidad de tramos.
- Cantidad de transferencias.
- Estación de transferencia, cuando corresponde.
- Clasificación de hora pico.
- Tiempo estimado.
- Gráficas del modelo y del recorrido.

### 6.2. Dataset incorporado

El dataset contiene 40 escenarios simulados de viaje.

Las variables utilizadas son:

| Variable | Descripción |
|---|---|
| `tramos` | Cantidad de desplazamientos entre estaciones |
| `transferencias` | Cantidad de cambios entre líneas |
| `hora_pico` | Indica si el recorrido se realiza en hora pico |
| `tiempo_minutos` | Tiempo total simulado del viaje |

La variable objetivo es `tiempo_minutos`.

Los tiempos son simulados y no representan los horarios, frecuencias ni tiempos oficiales del Metro de Medellín.

---

## 7. Entrenamiento y evaluación supervisada

Los dos componentes dividen los datos en:

- Conjunto de entrenamiento.
- Conjunto de prueba.

La división se realiza con `train_test_split`. Posteriormente, el modelo se entrena con `LinearRegression`.

Para evaluar su comportamiento se utilizan:

- **Error absoluto medio:** representa la diferencia promedio entre los valores reales del dataset y las predicciones.
- **Coeficiente de determinación:** permite observar qué proporción de la variación de los datos es explicada por el modelo.

Estas métricas se utilizan como una referencia académica. Debido a que los datasets son simulados y tienen una cantidad limitada de registros, no representan una validación suficiente para utilizar los modelos en situaciones reales.

---

# Actividad 4 Métodos de aprendizaje no supervisado

## 8. Aprendizaje no supervisado

El aprendizaje no supervisado analiza datos que no contienen una categoría de respuesta previamente asignada.

En esta actividad se utiliza el algoritmo **K-Means**, que organiza los registros en grupos según la similitud de sus características.

El procedimiento general es:

1. Preparar las variables numéricas.
2. Estandarizar los datos mediante `StandardScaler`.
3. Entrenar K-Means con tres grupos.
4. Analizar las características promedio de cada grupo.
5. Interpretar los grupos después del entrenamiento.
6. Asignar una nueva consulta al grupo más similar.
7. Representar los resultados mediante gráficas.

Los nombres asignados a los grupos sirven para facilitar su interpretación. No corresponden a etiquetas utilizadas durante el entrenamiento.

---

## 9. Sistema de condiciones de ruta mediante K-Means

### 9.1. Descripción

Este sistema analiza patrones en diferentes escenarios de viaje.

El usuario selecciona una ruta e ingresa:

- Hora de salida.
- Estado del clima.
- Medio de transporte.
- Minutos caminando antes del transporte.
- Minutos caminando después del transporte.

El modelo asigna la nueva consulta al grupo con características más similares.

Los grupos se interpretan posteriormente como:

- `PATRÓN FAVORABLE`
- `PATRÓN INTERMEDIO`
- `PATRÓN EXIGENTE`

Estas denominaciones se establecen después del entrenamiento a partir de las características promedio de cada grupo. No son categorías proporcionadas previamente a K-Means.

### 9.2. Dataset incorporado

El dataset contiene 18 viajes simulados.

Las variables utilizadas son:

| Variable | Descripción |
|---|---|
| `hora` | Hora de salida del viaje |
| `lluvia` | Indica si está lloviendo |
| `caminar_antes` | Minutos caminados antes del transporte |
| `caminar_despues` | Minutos caminados después del transporte |
| `metro` | Indica si el medio seleccionado es el Metro |
| `zona_industrial` | Indica si la ruta pasa por una zona industrial |

El dataset no contiene etiquetas de riesgo ni categorías de respuesta. K-Means descubre los grupos por similitud.

### 9.3. Resultados gráficos

El sistema genera dos gráficas.

#### Gráfica 1. Grupos de viajes identificados mediante K-Means

La gráfica presenta:

- Eje X: hora de salida.
- Eje Y: minutos totales de caminata.
- Colores: grupos descubiertos por K-Means.
- Estrella negra: viaje consultado por el usuario.

#### Gráfica 2. Cantidad de viajes por grupo

Esta gráfica muestra cuántos registros del dataset fueron asignados a cada grupo.

Los grupos representan patrones de similitud entre los viajes. No corresponden a una medición real de seguridad ni a una predicción de incidentes.

---

## 10. Sistema del Metro mediante A* y K-Means

### 10.1. Descripción

Este componente conserva la búsqueda de rutas entre estaciones de las líneas A y B del Metro de Medellín.

El algoritmo A* calcula el recorrido con la menor cantidad de tramos. Después, K-Means analiza las características de la ruta y la asigna al grupo con el que presenta mayor similitud.

Las variables analizadas son:

- Cantidad de tramos.
- Cantidad de transferencias.
- Condición de hora pico.

Después del entrenamiento, los grupos se interpretan como:

- `PATRÓN DE RECORRIDO CORTO`
- `PATRÓN DE RECORRIDO INTERMEDIO`
- `PATRÓN DE RECORRIDO EXTENSO`

La interpretación se realiza después de que K-Means ha creado los grupos.

### 10.2. Funcionamiento

El usuario debe:

1. Seleccionar la estación de inicio.
2. Seleccionar la estación de destino.
3. Ingresar la hora de salida.
4. Revisar la ruta calculada por A*.
5. Consultar el grupo asignado por K-Means.
6. Revisar las gráficas generadas.

El resultado presenta:

- Inicio del recorrido.
- Destino del recorrido.
- Mejor ruta encontrada.
- Cantidad de tramos.
- Cantidad de transferencias.
- Condición de hora pico.
- Grupo identificado.
- Patrón encontrado.
- Explicación general del resultado.

### 10.3. Resultados gráficos

El sistema genera dos gráficas.

#### Gráfica 1. Grupos descubiertos por K-Means

La gráfica utiliza:

- Eje X: cantidad de tramos.
- Eje Y: cantidad de transferencias.
- Colores: grupos descubiertos.
- Estrella negra: recorrido consultado.

#### Gráfica 2. Cantidad de viajes por grupo

La gráfica muestra el número de registros asignados a cada patrón.

Los grupos representan similitudes entre recorridos simulados y no mediciones oficiales de duración, congestión o calidad del servicio.

---

## 11. Librerías utilizadas

Los componentes utilizan las siguientes librerías:

```python
import heapq
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
```

### Función de las librerías

| Librería | Función |
|---|---|
| `heapq` | Implementar la cola de prioridad utilizada por A* |
| `pandas` | Crear y organizar los datasets |
| `matplotlib` | Generar las gráficas |
| `LinearRegression` | Entrenar los modelos supervisados |
| `KMeans` | Crear grupos de registros similares |
| `StandardScaler` | Estandarizar las variables antes del agrupamiento |
| `train_test_split` | Dividir los datos supervisados |
| `mean_absolute_error` | Calcular el error absoluto medio |
| `r2_score` | Calcular el coeficiente de determinación |

`heapq` pertenece a la biblioteca estándar de Python. Las demás librerías normalmente se encuentran disponibles en Google Colab.

---

## 12. Instrucciones para ejecutar los códigos en Google Colab

### Paso 1. Abrir Google Colab

Ingrese a [Google Colab](https://colab.research.google.com/) e inicie sesión con una cuenta de Google.

### Paso 2. Abrir el cuaderno

Utilice el enlace del cuaderno correspondiente o cree un nuevo cuaderno mediante:

```text
Archivo → Nuevo cuaderno
```

### Paso 3. Cargar el archivo

Si el código se encuentra en un archivo `.py`, puede subirlo desde el panel de archivos de Colab.

También puede ejecutar:

```python
from google.colab import files
files.upload()
```

### Paso 4. Instalar las librerías

Si alguna librería no estuviera disponible, ejecute:

```python
!pip install pandas scikit-learn matplotlib
```

### Paso 5. Ejecutar el archivo

Después de subir el archivo, ejecute:

```python
%run nombre_del_archivo.py
```

Debe reemplazar `nombre_del_archivo.py` por el nombre real del archivo que desea ejecutar.

### Paso 6. Ingresar los datos

Escriba la información solicitada respetando:

- Los nombres de las rutas o estaciones mostrados.
- Las opciones válidas de clima.
- Los medios de transporte disponibles.
- El intervalo de hora entre 0 y 23.
- Los valores numéricos solicitados.

### Paso 7. Revisar los resultados

Los sistemas presentan los resultados en la consola de Colab. Los componentes también generan gráficas debajo de la salida textual.

---

## 13. Datos utilizados

Los datasets se encuentran incorporados directamente en los códigos.

No es necesario:

- Descargar archivos CSV.
- Conectarse a una base de datos.
- Consultar una API.
- Cargar información externa.
- Registrar credenciales.

Los datos fueron construidos con fines académicos para representar diferentes condiciones de viaje y permitir el entrenamiento de los modelos.

---

## 14. Alcance y limitaciones

Los desarrollos son simulaciones académicas.

Por lo tanto:

- No consultan información en tiempo real.
- No utilizan registros oficiales del Metro de Medellín.
- No predicen delitos ni incidentes reales.
- No determinan la seguridad real de una persona.
- No reemplazan aplicaciones oficiales de transporte.
- No deben utilizarse para tomar decisiones de movilidad o seguridad.
- Los horarios definidos en los códigos se utilizan únicamente para aplicar las reglas de la simulación.
- Las estimaciones dependen de datasets pequeños y simulados.

Para implementar estos modelos en un contexto real sería necesario contar con datos históricos suficientes, actualizados, confiables y representativos.

---

## 15. Pruebas

Las pruebas comprueban:

- Ejecución correcta de los modelos.
- Selección de rutas válidas.
- Validación de entradas.
- Cálculo de rutas mediante A*.
- Predicciones de los modelos supervisados.
- Asignación de grupos mediante K-Means.
- Generación de resultados textuales.
- Presentación de las gráficas.

Las evidencias de las pruebas se encuentran documentadas en los entregables de las actividades 3 y 4.

---

## 16. Integrantes

| Integrante | Aporte |
|---|---|
| Dayana Monsalve | Desarrollo, pruebas y documentación de los componentes del proyecto |
| Daniela Camacho | Desarrollo, pruebas y documentación de los componentes del proyecto |

El historial de commits permite consultar los aportes registrados en el repositorio.

---

## 17. Referencia académica

Palma Méndez, J. T., Marín Morales, R., Rincón Zamorano, M., y colaboradores. (2008). *Inteligencia artificial: métodos, técnicas y aplicaciones*. McGraw-Hill España.

---

## 18. Nota final

Este repositorio fue desarrollado como parte de las actividades académicas del curso de Inteligencia Artificial.

Los resultados permiten observar cómo los métodos de aprendizaje supervisado y no supervisado pueden integrarse con sistemas basados en reglas y algoritmos de búsqueda. Las estimaciones y agrupaciones deben interpretarse dentro del alcance educativo del proyecto.

