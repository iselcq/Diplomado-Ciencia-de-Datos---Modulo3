"""
Algoritmos basados en distancia

1. El sistema deber permitir definir el tamaño de los vectores de entrada y salida
2. El sistema debe permitir introducir los valores para los vectores de entrada
y salida, para que pueda aprender dicha Asociación
3. El sistema debe permitir entrenar a partir de una base de datos almacenada en un archivo
de texto plano
"""
import csv

def verComoMatriz(matriz):
    cont=0
    while cont<len(matriz):
        print (matriz[cont])
        cont=cont+1

#  ≽(•⩊ •マ≼ CONFIGURACIÓN ≽(•⩊ •マ≼

tamañoEntrada = int(input("Tamaño del vector de ENTRADA: "))
tamañoSalida = int(input("Tamaño del vector de SALIDA: "))
k = int(input("Número de vecinos a considerar (K): "))
print("✩₊˚.⋆☾⋆⁺₊✧")
print(f"Configuración: X= {tamañoEntrada} dimensiones, Y= {tamañoSalida} dimensiones, K= {k}")

#  ≽(•⩊ •マ≼ FUNCIÓNES AUXILIARES ≽(•⩊ •マ≼

def convertir_a_float(vector):
    cont = 0
    resultado = []
    while cont < len(vector):
        resultado.append(float(vector[cont]))
        cont = cont + 1
    return resultado

def distancia(V1, V2, tipoDistancia):
    cont = 0
    suma = 0
    while cont < len(V1):
        if tipoDistancia == "1":  # Euclidiana
            diferencia = V1[cont] - V2[cont]
            suma = suma + (diferencia ** 2)
        else:  # Manhattan
            diferencia = V1[cont] - V2[cont]
            if diferencia < 0:
                diferencia = -diferencia
            suma = suma + diferencia
        cont = cont + 1
    if tipoDistancia == "1":
        suma = suma ** .5
    return suma

def convertir_a_float_vector(vector):
    cont = 0
    resultado = []
    while cont < len(vector):
        resultado.append(float(vector[cont]))
        cont = cont + 1
    return resultado

#  ≽(•⩊ •マ≼ ASOCIACIONES: almacenar pares [X, Y] ≽(•⩊ •マ≼

asociaciones = []  # cada elemento será [vectorX, vectorY]

def cargar_manual():
    numMuestras = int(input("Número de muestras a introducir: "))
    cont = 0
    while cont < numMuestras:
        print(f"Muestra {cont}")
        textoX = input(f"Introduce el vector X ({tamañoEntrada} valores separados por coma): ")
        vectorX = textoX.split(",")
        while len(vectorX) != tamañoEntrada:
            print(f"Error: se requieren exactamente {tamañoEntrada} valores.")
            vectorX = input("Introduce de nuevo el vector X: ").split(",")
        vectorX = convertir_a_float(vectorX)

        textoY = input(f"Introduce el vector Y ({tamañoSalida} valores separados por coma): ")
        vectorY = textoY.split(",")
        while len(vectorY) != tamañoSalida:
            print(f"Error: se requieren exactamente {tamañoSalida} valores.")
            vectorY = input("Introduce de nuevo el vector Y: ").split(",")
        vectorY = convertir_a_float(vectorY)

        asociaciones.append([vectorX, vectorY])
        cont = cont + 1

def cargar_archivo():
    nombreArchivo = input("Nombre del archivo de texto plano (sin extensión): ")
    if not nombreArchivo.endswith('.csv'):
        nombreArchivo = nombreArchivo + '.csv'
    separador = input("Carácter separador de datos (ej. coma: ','): ")

    with open(nombreArchivo, 'r', encoding='utf-8') as archivo:
        lector = csv.reader(archivo, delimiter=separador)
        for fila in lector:
            if not fila:
                continue
            vectorX = []
            cont = 0
            while cont < tamañoEntrada:
                vectorX.append(float(fila[cont]))
                cont = cont + 1
            vectorY = []
            cont = 0
            while cont < tamañoSalida:
                vectorY.append(fila[tamañoEntrada + cont])
                cont = cont + 1
            asociaciones.append([vectorX, vectorY])

print("✩₊˚.⋆☾⋆⁺₊✧")
print("(˶ᵔᗜᵔ˶)ﾉﾞ Origen de los datos: (˶ᵔᗜᵔ˶)ﾉﾞ")
print("  1. Introducir manualmente")
print("  2. Cargar desde archivo de texto plano")
opcion = input("Opción: ")

if opcion == "1":
    cargar_manual()
elif opcion == "2":
    cargar_archivo()
else:
    print("Opción no válida.")

print("✩₊˚.⋆☾⋆⁺₊✧")
print("Asociaciones aprendidas:")
verComoMatriz(asociaciones)

#   ≽(•⩊ •マ≼ APRENDIZAJE: ≽(•⩊ •マ≼
print("✩₊˚.⋆☾⋆⁺₊✧")
print('APRENDIZAJE')

porcentajeEntrenamiento = float(input("(˶ᵔᗜᵔ˶)ﾉﾞ Introduce el porcentaje de entrenamiento(0-1):   (˶ᵔᗜᵔ˶)ﾉﾞ"))
numMuestrasEntrenamiento = int(porcentajeEntrenamiento*len(asociaciones))
print ("Cantidad de muestras entrenamiento: ", numMuestrasEntrenamiento)

print ('Porcentaje prueba: ', 1-porcentajeEntrenamiento)
numMuestrasPrueba = int((1-porcentajeEntrenamiento)*len(asociaciones))
print ("Cantidad de muestras de prueba: ", numMuestrasPrueba)

asociacionAprendida = []
renglonesAAprender = []

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
            # print('if ', clase, ' = ', claseCD )
            if claseCD == clase:
                indicesAprender.append(indice)
                cont = cont + 1
                if cont == numeroNecesario[i]:
                    ultimosIndice.append(indice)
            indice = indice + 1

    return indicesAprender, ultimosIndice

clases = obtenerClases(asociaciones)
ultimosIndice = []
indicesIniciales = [0] * len(clases)

# Muestras por clase del conjunto completo
muestrasPorClase = int(len(asociaciones) / len(clases))

# Entrenamiento y Prueba: porcentaje aplicado a cada clase (redondeo hacia abajo)
numEntrenamientoPorClase = int(muestrasPorClase * porcentajeEntrenamiento)
numPruebaPorClase = muestrasPorClase - numEntrenamientoPorClase

renglonesAprender, ultimosIndice = obtenerIndices(
    asociaciones, clases,
    [numEntrenamientoPorClase] * len(clases), indicesIniciales)

renglonesProbar, ultimosIndice = obtenerIndices(
    asociaciones, clases,
    [numPruebaPorClase] * len(clases), ultimosIndice)

print("✩₊˚.⋆☾⋆⁺₊✧")
print('Últimos indices')
print(ultimosIndice)

cont = 0
asociacionAprendida = []
while cont < len(renglonesAprender):
    asociacionAprendida.append(asociaciones[renglonesAprender[cont]])
    cont = cont + 1

print ('Asociaciones aprendidas')
verComoMatriz(asociacionAprendida)
print ('NumAprendidos=',len(asociacionAprendida))

# ≽(•⩊ •マ≼  APRENDIZAJE: Clasificador Minima Distancia ≽(•⩊ •マ≼

#calcular el centroide de cada clase)¿
def calcularCentroide(vectores):
    centroide = []
    cont = 0
    while cont < tamañoEntrada:
        centroide.append(0)
        cont = cont + 1

    cont = 0
    while cont < len(vectores):
        cont2 = 0
        while cont2 < tamañoEntrada:
            centroide[cont2] = centroide[cont2] + vectores[cont][cont2]
            cont2 = cont2 + 1
        cont = cont + 1

    cont2 = 0
    while cont2 < tamañoEntrada:
        centroide[cont2] = centroide[cont2] / len(vectores)
        cont2 = cont2 + 1

    return centroide

def aprenderMinimaDistancia():
    centroides = []
    ind = 0
    while ind < len(clases):
        vectoresClase = []
        cont = 0
        while cont < len(asociacionAprendida):
            if asociacionAprendida[cont][1][0] == clases[ind]:
                vectoresClase.append(convertir_a_float_vector(asociacionAprendida[cont][0]))
            cont = cont + 1
        centroide = calcularCentroide(vectoresClase)
        centroides.append([clases[ind], centroide])
        ind = ind + 1
    return centroides

print("✩₊˚.⋆☾⋆⁺₊✧")
print('APRENDIZAJE POR MÍNIMA DISTANCIA')
centroides = aprenderMinimaDistancia()
print('Centroides aprendidos por clase:')
verComoMatriz(centroides)

# ≽(•⩊ •マ≼ RECUPERACION: KNN ≽(•⩊ •マ≼
def clasificarKNN(VZ):
    cont = 0
    distancias = []
    while cont < len(asociacionAprendida):
        VX = convertir_a_float_vector(asociacionAprendida[cont][0])
        dist = distancia(VX, VZ, tipoDistancia)
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

# ≽(•⩊ •マ≼ RECUPERACION: Mínima distancia  ≽(•⩊ •マ≼
def clasificarMinimaDistancia(VZ):
    cont = 0
    distancias = []
    while cont < len(centroides):
        dist = distancia(centroides[cont][1], VZ, tipoDistancia)
        distancias.append([dist, centroides[cont][0]])
        cont = cont + 1

    distanciasOrdenadas = sorted(distancias, key=lambda x: x[0])
    return distanciasOrdenadas[0][1]

print("✩₊˚.⋆☾⋆⁺₊✧")
print('RECUPERACIÓN')

# Elegir el método a usar
print('(˶ᵔᗜᵔ˶)ﾉﾞSelecciona el clasificador: (˶ᵔᗜᵔ˶)ﾉﾞ')
print('  1. KNN (K vecinos más cercanos)')
print('  2. Mínima distancia')
metodo = input("Opción: ")

# Elegir la distancia a usar (compartida por ambos métodos)
print('(˶ᵔᗜᵔ˶)ﾉﾞSelecciona el tipo de distancia: (˶ᵔᗜᵔ˶)ﾉﾞ')
print('  1. Euclidiana')
print('  2. Manhattan')
tipoDistancia = input("Opción: ")

def clasificar(VZ):
    if metodo == "1":
        return clasificarKNN(VZ)
    else:
        return clasificarMinimaDistancia(VZ)

# Evaluar con el conjunto de prueba
cont = 0
aciertos = 0
while cont < len(renglonesProbar):
    XPrueba = asociaciones[renglonesProbar[cont]][0]
    YReal = asociaciones[renglonesProbar[cont]][1][0]
    XPruebaFloat = convertir_a_float_vector(XPrueba)
    prediccion = clasificar(XPruebaFloat)
    if prediccion == YReal:
        aciertos = aciertos + 1
    cont = cont + 1

precision = (aciertos / len(renglonesProbar)) * 100
print('Aciertos:', aciertos, 'de', len(renglonesProbar))
print('Precisión: ', precision, '%')

# Clasificar un vector desconocido Z ingresado por el usuario
print("(˶ᵔᗜᵔ˶)ﾉﾞSelecciona la opción: (˶ᵔᗜᵔ˶)ﾉﾞ")
print("  1. Clasificar un vector Z desconocido")
print('  2. Salir')
opcionZ = input("Opción: ")

if opcionZ == "1":
    textoz = input(f"Introduce el vector desconocido Z ({tamañoEntrada} valores separados por coma): ")
    vectorZ = textoz.split(",")
    VZFloat = convertir_a_float_vector(vectorZ)
    clasePredicha = clasificar(VZFloat)
    print("El vector Z pertenece a la clase:", clasePredicha)
else:
    print(" (˶ᵔᗜᵔ˶)ﾉﾞFin del programa. (˶ᵔᗜᵔ˶)ﾉﾞ")

#  Vz = 6.5, 3.0, 5.8, 2.2
