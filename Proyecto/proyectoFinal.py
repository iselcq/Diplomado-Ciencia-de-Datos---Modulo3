"""
Proyecto Machine Learning - Clasificación
Considerar la base de datos Iris-plant en la p[agina de UCI machin e learning
https://archive.ics.uci.edu/

"""

import pandas as pd
import numpy as np
import csv

#1. Debe describir cada uno de los atributos al momento. (˵ ¬ᴗ¬˵)
# ≽(•⩊ •マ≼ Cargar base de datos ≽(•⩊ •マ≼
nombreArchivo = "irisDB"
separador = ","

def verComoMatriz(matriz):
    cont = 0
    while cont < len(matriz):
        print(matriz[cont])
        cont = cont + 1

def cargar_csv(ruta_csv, separador):
    datos = []
    with open(ruta_csv, 'r', encoding='utf-8') as archivo:
        lector = csv.reader(archivo, delimiter=separador)
        ### Pre-procesamiento:no tomar en cuenta fila vacia de la base de datos
        for fila in lector:
            if not fila:
                continue
            datos.append(fila)
    return datos

if not nombreArchivo.endswith('.csv'):
    nombreArchivo = nombreArchivo + '.csv'

matrizDatos = cargar_csv(nombreArchivo, separador)

# ≽(•⩊ •マ≼ Analizar cada atributo para decidir su tipo  ≽(•⩊ •マ≼

index_cuantitativos = []
index_cualitativos = []
float_cuantitativos = []
str_cualitativos = []

numAtributos = len(matrizDatos[0])

for j in range(numAtributos):
    columna = []
    for fila in matrizDatos:
        columna.append(fila[j])
    try:
        cuantitativas = [float(x) for x in columna]
        index_cuantitativos.append(j)
        float_cuantitativos.append(cuantitativas)
        print(f"Atributo {j}: CUANTITATIVO")
    except ValueError:
        index_cualitativos.append(j)
        str_cualitativos.append(columna)
        print(f"Atributo {j}: CUALITATIVO ")

# ≽(•⩊ •マ≼ Análisis según el tipo  ≽(•⩊ •マ≼

#Cualitativo cateogorias
for j in index_cualitativos:
    categorias = list(set(fila[j] for fila in matrizDatos))
    print(f"Categorías del atributo {j}: {categorias}")

# Cuantitativo mínimo, máximo, promedio, desviación estándar
matrizCuantitativas = np.array(float_cuantitativos, dtype=float)

for k in range(matrizCuantitativas.shape[0]):
    indice = index_cuantitativos[k]
    promedio = matrizCuantitativas[k].mean()
    minimo = matrizCuantitativas[k].min()
    maximo = matrizCuantitativas[k].max()
    dev_std = matrizCuantitativas[k].std()
    print(f"Atributo {indice} ---> "
          f"Promedio= {promedio:.4f},",
          f"Min= {minimo:.4f},",
          f"Max= {maximo:.4f},",
          f"Desviación Estándar= {dev_std:.4f}.")

#Falta tomas solo 10 atributos si son mas de 10

#2. Definir los atributos del vector de entrada X y de salida (clase)Y (˵ ¬ᴗ¬˵)

# ≽(•⩊ •マ≼ CONSTRUIR ASOCIACIONES [X, Y] ≽(•⩊ •マ≼
indicesEntrada = index_cuantitativos
indicesSalida = index_cualitativos

def construirX(fila):
    vectorX = []
    cont = 0
    while cont < len(indicesEntrada):
        vectorX.append(float(fila[indicesEntrada[cont]]))
        cont = cont + 1
    return vectorX

def construirY(fila):
    vectorY = []
    cont = 0
    while cont < len(indicesSalida):
        vectorY.append(fila[indicesSalida[cont]])
        cont = cont + 1
    return vectorY

asociaciones = []
cont = 0
while cont < len(matrizDatos):
    asociaciones.append([construirX(matrizDatos[cont]), construirY(matrizDatos[cont])])
    cont = cont + 1

print("Asociaciones [X, Y] construidas:")
verComoMatriz(asociaciones)

def obtenerClases(asociaciones):
    clases = []
    cont = 0
    while cont < len(asociaciones):
        etiqueta = asociaciones[cont][1][0]
        if etiqueta not in clases:
            clases.append(etiqueta)
        cont = cont + 1
    return clases

vectorClases = obtenerClases(asociaciones)
print(f"Clases (Y): {vectorClases}")

# ≽(•⩊ •マ≼ Estadísticas (min, max, std, mean) de cada atributo de entrada X ≽(•⩊ •マ≼

nombresAtributosX = ["LS", "AS", "LP", "AP"]

matrizX = np.array([a[0] for a in asociaciones], dtype=float)
for j in range(matrizX.shape[1]):
    columna = matrizX[:, j]
    print(f"{nombresAtributosX[j]} --> Promedio = {columna.mean():.4f}, "
          f"Min= {columna.min():.4f}, "
          f"Max= {columna.max():.4f}, "
          f"Desviación Estándar= {columna.std():.4f}.")

# ≽(•⩊ •マ≼ Vector de salida Y (clase) por cada muestra ≽(•⩊ •マ≼
vectorY = np.array([a[1][0] for a in asociaciones])

#3. Pre-procesamiento a la base de datos, describirlo (˵ ¬ᴗ¬˵)
    # linea 27

#4. Obtener las métricas de recuperación (% aciertos) y error del clasificador K-NN
#   usando el metodo de validacion Train and Test

# ≽(•⩊ •マ≼ FUNCIONES AUXILIARES DE K-NN ≽(•⩊ •マ≼

def distancia(V1, V2):
    # Distancia euclidiana
    cont = 0
    suma = 0
    while cont < len(V1):
        diferencia = V1[cont] - V2[cont]
        suma = suma + (diferencia ** 2)
        cont = cont + 1
    return suma ** .5

def clasificarKNN(VZ, asociacionAprendida, k):
    cont = 0
    distancias = []
    while cont < len(asociacionAprendida):
        VX = asociacionAprendida[cont][0]
        dist = distancia(VX, VZ)
        distancias.append([dist, cont])
        cont = cont + 1

    distanciasOrdenadas = sorted(distancias, key=lambda x: x[0])

    #Tomar las K distancias más pequeñas (los K vecinos más cercanos)
    cont = 0
    clasesCercanas = []
    while cont < k:
        clasesCercanas.append(asociacionAprendida[distanciasOrdenadas[cont][1]][1][0])
        cont = cont + 1

    #Votación
    conteo = [0] * len(vectorClases)
    cont = 0
    while cont < len(clasesCercanas):
        j = 0
        while j < len(vectorClases):
            if clasesCercanas[cont] == vectorClases[j]:
                conteo[j] = conteo[j] + 1
            j = j + 1
        cont = cont + 1

    cont = 0
    maximo = 0
    indiceMaximo = 0
    while cont < len(conteo):
        if maximo < conteo[cont]:
            maximo = conteo[cont]
            indiceMaximo = cont
        cont = cont + 1

    return vectorClases[indiceMaximo]

# ≽(•⩊ •マ≼ METODO DE VALIDACIÓN: TRAIN AND TEST (balanceado por clase) ≽(•⩊ •マ≼
def validarTrainAndTest(asociaciones, clases, porcentajeEntrenamiento):
    numPorClase = int(len(asociaciones) / len(clases))
    numEntrenamientoPorClase = int(numPorClase * porcentajeEntrenamiento)
    numPruebaPorClase = numPorClase - numEntrenamientoPorClase

    indicesIniciales = [0] * len(clases)

    def obtenerIndices(numeroNecesario, indicesIniciales):
        indicesAprender = []
        ultimosIndice = []
        for i in range(len(clases)):
            indice = indicesIniciales[i]
            clase = clases[i]
            cont = 0
            while cont < numeroNecesario[i]:
                claseCD = asociaciones[indice][1][0]
                if claseCD == clase:
                    indicesAprender.append(indice)
                    cont = cont + 1
                    if cont == numeroNecesario[i]:
                        ultimosIndice.append(indice)
                indice = indice + 1
        return indicesAprender, ultimosIndice

    renglonesEntrenar, ultimosIndice = obtenerIndices(
        [numEntrenamientoPorClase] * len(clases), indicesIniciales)

    for i in range(len(ultimosIndice)):
        ultimosIndice[i] = ultimosIndice[i] + 1

    renglonesProbar, _ = obtenerIndices(
        [numPruebaPorClase] * len(clases), ultimosIndice)

    # ENTRENAR: K-NN almacenando las asociaciones de entrenamiento
    asociacionAprendida = []
    cont = 0
    while cont < len(renglonesEntrenar):
        asociacionAprendida.append(asociaciones[renglonesEntrenar[cont]])
        cont = cont + 1

    return asociacionAprendida, renglonesProbar

def evaluarKNN(asociaciones, asociacionAprendida, renglonesProbar, k):
    aciertos = 0
    cont = 0
    while cont < len(renglonesProbar):
        XPrueba = asociaciones[renglonesProbar[cont]][0]
        YReal = asociaciones[renglonesProbar[cont]][1][0]
        prediccion = clasificarKNN(XPrueba, asociacionAprendida, k)
        if prediccion == YReal:
            aciertos = aciertos + 1
        cont = cont + 1

    totalPrueba = len(renglonesProbar)
    recuperacion = (aciertos / totalPrueba) * 100
    error = ((totalPrueba - aciertos) / totalPrueba) * 100
    return aciertos, totalPrueba, recuperacion, error

# ≽(•⩊ •マ≼ EVALUAR K-NN CON TRAIN AND TEST ≽(•⩊ •マ≼

porcentajeEntrenamiento = 0.7
k = int(input("Número de vecinos a considerar (K): "))

asociacionAprendida, renglonesProbar = validarTrainAndTest(
    asociaciones, vectorClases, porcentajeEntrenamiento)

aciertos, totalPrueba, recuperacion, error = evaluarKNN(
    asociaciones, asociacionAprendida, renglonesProbar, k)

print("✩₊˚.⋆☾⋆⁺₊✧")
print("MÉTRICAS DE RECUPERACIÓN DEL CLASIFICADOR K-NN")
print(f"TRAIN AND TEST ({int(porcentajeEntrenamiento*100)}% entrenamiento, K={k})")
print("✩₊˚.⋆☾⋆⁺₊✧")
print(f"Muestras de entrenamiento: {len(asociacionAprendida)}")
print(f"Muestras de prueba:        {totalPrueba}")
print(f"Aciertos: {aciertos} de {totalPrueba}")
print(f"Recuperación (% aciertos): {recuperacion:.2f}%")
print(f"Error: {error:.2f}%")
print("✩₊˚.⋆☾⋆⁺₊✧")


#5. Obtener las métricas de recuperación (% aciertos) y error del clasificador
    #Mínima distancia

# ≽(•⩊ •マ≼ FUNCIONES DEL CLASIFICADOR DE MÍNIMA DISTANCIA ≽(•⩊ •マ≼

def calcularCentroide(vectores):
    # Promedio de cada dimensión de los vectores de una clase
    centroide = []
    dim = 0
    while dim < len(vectores[0]):
        suma = 0
        cont = 0
        while cont < len(vectores):
            suma = suma + vectores[cont][dim]
            cont = cont + 1
        centroide.append(suma / len(vectores))
        dim = dim + 1
    return centroide

def aprenderMinimaDistancia(asociacionAprendida, clases):
    # Por cada clase,  vectores X y calcular el centroide
    centroides = []
    indiceClase = 0
    while indiceClase < len(clases):
        clase = clases[indiceClase]
        vectoresClase = []
        cont = 0
        while cont < len(asociacionAprendida):
            if asociacionAprendida[cont][1][0] == clase:
                vectoresClase.append(asociacionAprendida[cont][0])
            cont = cont + 1
        centroide = calcularCentroide(vectoresClase)
        centroides.append([clase, centroide])
        indiceClase = indiceClase + 1
    return centroides

def clasificarMinimaDistancia(VZ, centroides):
    cont = 0
    distancias = []
    while cont < len(centroides):
        dist = distancia(centroides[cont][1], VZ)
        distancias.append([dist, centroides[cont][0]])
        cont = cont + 1

    distanciasOrdenadas = sorted(distancias, key=lambda x: x[0])
    return distanciasOrdenadas[0][1]

# ≽(•⩊ •マ≼ EVALUAR MÍNIMA DISTANCIA CON TRAIN AND TEST ≽(•⩊ •マ≼

asociacionAprendidaMD, renglonesProbarMD = validarTrainAndTest(
    asociaciones, vectorClases, porcentajeEntrenamiento)

centroides = aprenderMinimaDistancia(asociacionAprendidaMD, vectorClases)

print("✩₊˚.⋆☾⋆⁺₊✧")
print("CENTROIDES APRENDIDOS POR CLASE")
for centro in centroides:
    print(f"  {centro[0]}: {[round(x, 4) for x in centro[1]]}")

aciertosMD = 0
cont = 0
while cont < len(renglonesProbarMD):
    XPrueba = asociaciones[renglonesProbarMD[cont]][0]
    YReal = asociaciones[renglonesProbarMD[cont]][1][0]
    prediccion = clasificarMinimaDistancia(XPrueba, centroides)
    if prediccion == YReal:
        aciertosMD = aciertosMD + 1
    cont = cont + 1

totalPruebaMD = len(renglonesProbarMD)
recuperacionMD = (aciertosMD / totalPruebaMD) * 100
errorMD = ((totalPruebaMD - aciertosMD) / totalPruebaMD) * 100

print("✩₊˚.⋆☾⋆⁺₊✧")
print("MÉTRICAS DE RECUPERACIÓN DEL CLASIFICADOR DE MÍNIMA DISTANCIA")
print(f"TRAIN AND TEST ({int(porcentajeEntrenamiento*100)}% entrenamiento)")
print("✩₊˚.⋆☾⋆⁺₊✧")
print(f"Muestras de entrenamiento: {len(asociacionAprendidaMD)}")
print(f"Muestras de prueba:        {totalPruebaMD}")
print(f"Aciertos: {aciertosMD} de {totalPruebaMD}")
print(f"Recuperación (% aciertos): {recuperacionMD:.2f}%")
print(f"Error: {errorMD:.2f}%")
print("✩₊˚.⋆☾⋆⁺₊✧")


#6. Elegir dos de los atributos utilizando algún criterio
    ## Eliminar uno de los atributos elegidos y corroborar metodo de validacion
    ## Eliminar el otro de los atributos elegidos y corroborar metodo de validacion
    ## Eliminar los dos atributos elgidos y corroborar metodo de validacion
#7. Reemplaza las muestras y obten porc entajes de recuperacion y error (1-NN,3-NN y 5-NN)


