"""
PRÁCTICA DE LABORATORIO: MÉTODOS DE VALIDACIÓN
Train and Test

1. El sistema debe permitir cargar la base de datos con la cual se trabajará
2. Debe permitir especificar los atributos a utilizar para construir el vector de entrada X
3. Debe permitir especificar los atributos (columnas) a utilizar para construir el vector de salida Y
"""
import csv
import math

def verComoMatriz(matriz):
    cont = 0
    while cont < len(matriz):
        print(matriz[cont])
        cont = cont + 1

# ≽(•⩊ •マ≼ CARGAR BASE DE DATOS ≽(•⩊ •マ≼

def cargar_csv(ruta_csv, separador):
    datos = []
    with open(ruta_csv, 'r', encoding='utf-8') as archivo:
        lector = csv.reader(archivo, delimiter=separador)
        for fila in lector:
            if not fila:
                continue
            datos.append(fila)
    return datos

print("✩₊˚.⋆☾⋆⁺₊✧")
print('CARGAR BASE DE DATOS')
nombreArchivo = input("Nombre del archivo de texto plano (sin extensión): ")
if not nombreArchivo.endswith('.csv'):
    nombreArchivo = nombreArchivo + '.csv'
separador = input("Carácter separador de datos (ej. coma: ','): ")

matriz = cargar_csv(nombreArchivo, separador)
print("Base de datos cargada: (˶ᵔᗜᵔ˶)ﾉﾞ ")
verComoMatriz(matriz)
print('NumMuestras =', len(matriz))
numColumnas = len(matriz[0])
print('NumColumnas =', numColumnas)

# ≽(•⩊ •マ≼ INGRESAR ATRIBUTOS PARA EL VECTOR X DE ENTRADA ≽(•⩊ •マ≼

print("✩₊˚.⋆☾⋆⁺₊✧")
print('SELECCIÓN DE ATRIBUTOS DE ENTRADA (X)')
print('Columnas disponibles (índices 0 a', numColumnas - 1, '):')

def leerIndicesColumnas(mensaje):
    valido = False
    while not valido:
        texto = input(mensaje)
        try:
            indices = [int(i) for i in texto.split(",")]
        except ValueError:
            print('Error: introduce solo números separados por coma.')
            continue
        if any(i < 0 or i >= numColumnas for i in indices):
            print('Error: algún índice está fuera del rango 0 a', numColumnas - 1)
            continue
        valido = True
    return indices

indicesEntrada = leerIndicesColumnas("(˶ᵔᗜᵔ˶)ﾉﾞ Introduce los índices de columna para X (separados por coma): (˶ᵔᗜᵔ˶)ﾉﾞ ")
tamañoEntrada = len(indicesEntrada)

# ≽(•⩊ •マ≼ ESPECIFICAR ATRIBUTOS PARA EL VECTOR Y DE SALIDA ≽(•⩊ •マ≼

print("✩₊˚.⋆☾⋆⁺₊✧")
print('SELECCIÓN DE ATRIBUTOS DE SALIDA (Y)')
indicesSalida = leerIndicesColumnas("(˶ᵔᗜᵔ˶)ﾉﾞ Introduce los índices de columna para Y (separados por coma): (˶ᵔᗜᵔ˶)ﾉﾞ ")
tamañoSalida = len(indicesSalida)


# ≽(•⩊ •マ≼ CONSTRUIR ASOCIACIONES [X, Y] ≽(•⩊ •マ≼

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
while cont < len(matriz):
    asociaciones.append([construirX(matriz[cont]), construirY(matriz[cont])])
    cont = cont + 1

print("Asociaciones [X, Y] construidas:")
verComoMatriz(asociaciones)


# ≽(•⩊ •マ≼ TRAIN AND TEST (división balanceada por clase) ≽(•⩊ •マ≼

print("✩₊˚.⋆☾⋆⁺₊✧")
print('TRAIN AND TEST')

def leerPorcentaje():
    valido = False
    while not valido:
        texto = input("˶ᵔᗜᵔ˶)ﾉﾞ Introduce el porcentaje de entrenamiento(0-1): ˶ᵔᗜᵔ˶)ﾉﾞ ")
        try:
            p = float(texto)
        except ValueError:
            print('Error: introduce un número (ej. 0.7).')
            continue
        if p < 0 or p > 1:
            print('Error: el porcentaje debe estar entre 0 y 1.')
            continue
        valido = True
    return p

porcentajeEntrenamiento = leerPorcentaje()

def obtenerClases(asociaciones):
    clases = []
    cont = 0
    while cont < len(asociaciones):
        etiqueta = asociaciones[cont][1][0]
        if etiqueta not in clases:
            clases.append(etiqueta)
        cont = cont + 1
    return clases

def obtenerIndices(asociaciones, clases, numeroNecesario, indicesIniciales):
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

clases = obtenerClases(asociaciones)
indicesIniciales = [0] * len(clases)
ultimosIndice = []

muestrasPorClase = int(len(asociaciones) / len(clases))
numEntrenamientoPorClase = int(muestrasPorClase * porcentajeEntrenamiento)
numPruebaPorClase = muestrasPorClase - numEntrenamientoPorClase

renglonesEntrenar, ultimosIndice = obtenerIndices(
    asociaciones, clases,
    [numEntrenamientoPorClase] * len(clases), indicesIniciales)

renglonesProbar, ultimosIndice = obtenerIndices(
    asociaciones, clases,
    [numPruebaPorClase] * len(clases), ultimosIndice)

print('Renglones de entrenamiento:', len(renglonesEntrenar))
print('Renglones de prueba:', len(renglonesProbar))


# ≽(•⩊ •マ≼ ENTRENAR (KNN: almacenar las asociaciones de entrenamiento) ≽(•⩊ •マ≼

asociacionAprendida = []
cont = 0
while cont < len(renglonesEntrenar):
    asociacionAprendida.append(asociaciones[renglonesEntrenar[cont]])
    cont = cont + 1

print('Asociaciones aprendidas (entrenamiento):')
verComoMatriz(asociacionAprendida)
print('NumAprendidos =', len(asociacionAprendida))


# ≽(•⩊ •マ≼ EVALUAR CON EL CONJUNTO DE PRUEBA ≽(•⩊ •マ≼

print("✩₊˚.⋆☾⋆⁺₊✧")
print('EVALUACIÓN (KNN)')

def leerK(numAprendidos):
    valido = False
    while not valido:
        texto = input("(˶ᵔᗜᵔ˶)ﾉﾞ Número de vecinos a considerar (K): (˶ᵔᗜᵔ˶)ﾉﾞ ")
        try:
            valorK = int(texto)
        except ValueError:
            print('Error: introduce un número entero.')
            continue
        if valorK < 1:
            print('Error: K debe ser al menos 1.')
            continue
        if valorK > numAprendidos:
            print('Error: K no puede superar el número de muestras aprendidas (', numAprendidos, ').')
            continue
        valido = True
    return valorK

k = leerK(len(asociacionAprendida))

def distancia(V1, V2):
    cont = 0
    suma = 0
    while cont < len(V1):
        diferencia = V1[cont] - V2[cont]
        suma = suma + (diferencia ** 2)
        cont = cont + 1
    return suma ** .5

def clasificar(VZ):
    cont = 0
    distancias = []
    while cont < len(asociacionAprendida):
        dist = distancia(asociacionAprendida[cont][0], VZ)
        distancias.append([dist, cont])
        cont = cont + 1

    distanciasOrdenadas = sorted(distancias, key=lambda x: x[0])

    cont = 0
    clasesCercanas = []
    while cont < k:
        clasesCercanas.append(asociacionAprendida[distanciasOrdenadas[cont][1]][1][0])
        cont = cont + 1

    cont = 0
    conteo = [0] * len(clases)
    while cont < len(clasesCercanas):
        j = 0
        while j < len(clases):
            if clasesCercanas[cont] == clases[j]:
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

    return clases[indiceMaximo]

cont = 0
aciertos = 0
while cont < len(renglonesProbar):
    XPrueba = asociaciones[renglonesProbar[cont]][0]
    YReal = asociaciones[renglonesProbar[cont]][1][0]
    prediccion = clasificar(XPrueba)
    if prediccion == YReal:
        aciertos = aciertos + 1
    cont = cont + 1

errores = len(renglonesProbar) - aciertos
precision = (aciertos / len(renglonesProbar)) * 100
porcentajeError = (errores / len(renglonesProbar)) * 100
print('Aciertos:', aciertos, 'de', len(renglonesProbar))
print('Errores:', errores, 'de', len(renglonesProbar))
print('Precisión (Train and Test): ', precision, '%')
print('Porcentaje de error: ', porcentajeError, '%')
