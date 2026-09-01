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
          f"Promedio = {promedio:.4f},",
          f"Min = {minimo:.4f},",
          f"Max = {maximo:.4f},",
          f"Desviación Estándar = {dev_std:.4f}.")

#Falta tomar solo 10 atributos si son mas de 10

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

#4. Obtener las métricas de recuperación (% aciertos) y error del clasificador K-NN usando el metodo de validacion Train and Test


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


#5. Obtener las métricas de recuperación (% aciertos) y error del clasificador mínima distancia
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


#6. Elegir dos de los atributos utilizando algún criterio y corroborar con Train & Test (˵ ¬ᴗ¬˵)

print("6. ELIMINACIÓN DE ATRIBUTOS")

# Posiciones de los atributos de entrada
posXCompleta = [0, 1, 2, 3]
# Atributos a eliminar
atributosEliminar = [2, 3]  # LP y AP

# ≽(•⩊ •マ≼ FUNCIÓN PARA RE-CONSTRUIR ASOCIACIONES ≽(•⩊ •マ≼
def construirAsociacionesConMatriz(matriz, posX):
    nuevas = []
    cont = 0
    while cont < len(matriz):
        VX = []
        dim = 0
        while dim < len(posX):
            VX.append(float(matriz[cont][posX[dim]]))
            dim = dim + 1
        VY = [matriz[cont][4]]  # columna 4 = clase
        nuevas.append([VX, VY])
        cont = cont + 1
    return nuevas

# ≽(•⩊ •マ≼ FUNCIONES GENÉRICAS DE EVALUACIÓN ≽(•⩊ •マ≼
def evaluarKNNGenerico(asociacionesXT, clases, k):
    asociacionAprendida, renglonesProbar = validarTrainAndTest(
        asociacionesXT, clases, porcentajeEntrenamiento)
    aciertos, total, rec, err = evaluarKNN(
        asociacionesXT, asociacionAprendida, renglonesProbar, k)
    return rec, err

def evaluarMinimaDistanciaGenerico(asociacionesXT, clases):
    asociacionAprendida, renglonesProbar = validarTrainAndTest(
        asociacionesXT, clases, porcentajeEntrenamiento)
    centroides = aprenderMinimaDistancia(asociacionAprendida, clases)
    aciertos = 0
    cont = 0
    while cont < len(renglonesProbar):
        XPrueba = asociacionesXT[renglonesProbar[cont]][0]
        YReal = asociacionesXT[renglonesProbar[cont]][1][0]
        prediccion = clasificarMinimaDistancia(XPrueba, centroides)
        if prediccion == YReal:
            aciertos = aciertos + 1
        cont = cont + 1
    totalPrueba = len(renglonesProbar)
    rec = (aciertos / totalPrueba) * 100
    err = 100 - rec
    return rec, err

# ≽(•⩊ •マ≼ GENERAR LOS ESCENARIOS ≽(•⩊ •マ≼
# los 4 atributos. Luego se va quitando cada atributo elegido.

def descripcionEscenario(posX):
    # Devuelve un nombre legible a partir de las posiciones que quedan en X
    if posX == posXCompleta:
        return "Los 4 atributos (LS,AS,LP,AP)"
    quitados = [x for x in posXCompleta if x not in posX]
    nombresQuitados = [nombresAtributosX[x] for x in quitados]
    nombresQuedan = [nombresAtributosX[x] for x in posX]
    return f"Sin {' y '.join(nombresQuitados)} (deja {','.join(nombresQuedan)})"
escenarios = []

# 6 (ninguna eliminación)
escenarios.append((descripcionEscenario(posXCompleta), posXCompleta))
# 6a. quitar el primer atributo
posA = [x for x in posXCompleta if x not in [atributosEliminar[0]]]
escenarios.append((descripcionEscenario(posA), posA))
# 6b. quitar el segundo atributo
posB = [x for x in posXCompleta if x not in [atributosEliminar[1]]]
escenarios.append((descripcionEscenario(posB), posB))
# 6c. quitar ambos atributos
posC = [x for x in posXCompleta if x not in atributosEliminar]
escenarios.append((descripcionEscenario(posC), posC))

for nombre, posX in escenarios:
    asocEsc = construirAsociacionesConMatriz(matrizDatos, posX)
    recK, errK = evaluarKNNGenerico(asocEsc, vectorClases, k)
    recM, errM = evaluarMinimaDistanciaGenerico(asocEsc, vectorClases)

    print("✩₊˚.⋆☾⋆⁺₊✧")
    print("ESCENARIO:", nombre)
    print("K-NN")
    print(f"  Recuperación (% aciertos): {recK:.2f}%")
    print(f"  Error: {errK:.2f}%")
    print("Mínima Distancia")
    print(f"  Recuperación (% aciertos): {recM:.2f}%")
    print(f"  Error: {errM:.2f}%")


# ≽(•⩊ •マ≼ 7. SUSTITUIR 3 MUESTRAS DEL CONJUNTO DE APRENDIZAJE ≽(•⩊ •マ≼
print("✩₊˚.⋆☾⋆⁺₊✧")
print("7. SUSTITUCIÓN DE 3 MUESTRAS EN EL CONJUNTO DE APRENDIZAJE")

# Muestras nuevas que sustituirán a 3 muestras del aprendizaje
v1 = [50, 35, 14, 20, 'Iris-setosa']
v2 = [50, 35, 14, 20, 'Iris-setosa']
v3 = [50, 32, 13, 20, 'Iris-setosa']

muestrasNuevas = [v1, v2, v3]

# Obtener el conjunto de aprendizaje y de prueba (una sola división)
asociacionAprendida7, renglonesProbar7 = validarTrainAndTest(
    asociaciones, vectorClases, porcentajeEntrenamiento)

# ≽(•⩊ •マ≼ Función que reemplaza las 3 primeras muestras del aprendizaje ≽(•⩊ •マ≼
def sustituirMuestras(aprendidas, nuevas):
    modificadas = []
    cont = 0
    while cont < len(aprendidas):
        if cont < len(nuevas):
            # convertir v a asociación [[X],[Y]]
            VX = [float(nuevas[cont][0]), float(nuevas[cont][1]),
                  float(nuevas[cont][2]), float(nuevas[cont][3])]
            VY = [nuevas[cont][4]]
            modificadas.append([VX, VY])
        else:
            modificadas.append(aprendidas[cont])
        cont = cont + 1
    return modificadas

aprendidasModificadas = sustituirMuestras(asociacionAprendida7, muestrasNuevas)

# VERIFICACIÓN: las primeras muestras del aprendizaje antes y después
print("Antes de sustituir (primeras muestras del aprendizaje):")
for i in range(len(muestrasNuevas)):
    print("  ", asociacionAprendida7[i])
print("Después de sustituir (primeras muestras del aprendizaje):")
for i in range(len(muestrasNuevas)):
    print("  ", aprendidasModificadas[i])

# K-NN  sobre el aprendizaje modificado
def evaluarKNNSustitucion(asociaciones, renglonesProbar, aprendidas, valorK):
    aciertos = 0
    cont = 0
    while cont < len(renglonesProbar):
        XPrueba = asociaciones[renglonesProbar[cont]][0]
        YReal = asociaciones[renglonesProbar[cont]][1][0]
        prediccion = clasificarKNN(XPrueba, aprendidas, valorK)
        if prediccion == YReal:
            aciertos = aciertos + 1
        cont = cont + 1

    total = len(renglonesProbar)
    recuperacion = (aciertos / total) * 100
    error = 100 - recuperacion

    print("✩₊˚.⋆☾⋆⁺₊✧")
    print(f"K = {valorK}-NN (con muestras sustituidas)")
    print(f"  Muestras de prueba:        {total}")
    print(f"  Aciertos: {aciertos} de {total}")
    print(f"  Recuperación (% aciertos): {recuperacion:.2f}%")
    print(f"  Error: {error:.2f}%")

evaluarKNNSustitucion(asociaciones, renglonesProbar7, aprendidasModificadas, 1)
evaluarKNNSustitucion(asociaciones, renglonesProbar7, aprendidasModificadas, 3)
evaluarKNNSustitucion(asociaciones, renglonesProbar7, aprendidasModificadas, 5)

# Evaluación de MÍNIMA DISTANCIA sobre el aprendizaje con muestras sustituidas
def evaluarMinimaDistanciaSustitucion(asociaciones, renglonesProbar, aprendidas):
    centroides = aprenderMinimaDistancia(aprendidas, vectorClases)

    aciertos = 0
    cont = 0
    while cont < len(renglonesProbar):
        XPrueba = asociaciones[renglonesProbar[cont]][0]
        YReal = asociaciones[renglonesProbar[cont]][1][0]
        prediccion = clasificarMinimaDistancia(XPrueba, centroides)
        if prediccion == YReal:
            aciertos = aciertos + 1
        cont = cont + 1

    total = len(renglonesProbar)
    recuperacion = (aciertos / total) * 100
    error = 100 - recuperacion

    print("✩₊˚.⋆☾⋆⁺₊✧")
    print("Mínima Distancia (con muestras sustituidas)")
    print(f"  Muestras de prueba:        {total}")
    print(f"  Aciertos: {aciertos} de {total}")
    print(f"  Recuperación (% aciertos): {recuperacion:.2f}%")
    print(f"  Error: {error:.2f}%")

evaluarMinimaDistanciaSustitucion(asociaciones, renglonesProbar7, aprendidasModificadas)

print("✩₊˚.⋆☾⋆⁺₊✧")



