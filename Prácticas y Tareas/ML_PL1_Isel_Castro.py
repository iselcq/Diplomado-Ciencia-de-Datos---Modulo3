"""
Desarrolla un sistema que permita cargar conjuntos de datos almacenados en archivos de texto plano,
los cuales se utilizarán en los problemas de inteligencia artificial.
"""
import csv
import numpy as np


#1. Elegir el archivo de texto plano
nombreArchivo = input("Ingresa el nombre del archivo de texto plano (sin extensión): ")
if not nombreArchivo.endswith('.csv'):
    nombreArchivo = nombreArchivo + '.csv'

#2. Elegir el carácter separador de datos
separador = input("Ingresa el carácter separador de datos (ej. coma: ','): ")

#3-5. Cargar toda la información en una matriz de datos interna
def cargar_csv(ruta_csv, separador):
    datos=[]
    with open(ruta_csv, 'r', encoding='utf-8') as archivo:
        lector = csv.reader(archivo, delimiter=separador)
        for fila in lector:
            datos.append(fila)
    return datos

matrizDatos = cargar_csv(nombreArchivo, separador)

#4. Número de renglones con muestras
numRenglones = len(matrizDatos)
print(f"No. de Renglones con muestras: {numRenglones}")

#3. Número de atributos
numAtributos = len(matrizDatos[0])
print(f"No. de Atributos: {numAtributos}")

print("Matriz de datos interna:")
print(matrizDatos)

#6. Analizar cada atributo para decidir su tipo
index_cuantitativos = []
index_cualitativos = []
float_cuantitativos = []
str_cualitativos = []

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
        print(f"Atributo {j}: CUALITATIVO")

matrizCuantitativas = np.array(float_cuantitativos, dtype=float)

#7. Análisis según el tipo
print("ANÁLISIS DE ATRIBUTOS")
#a. Cualitativo: categorías
for j in index_cualitativos:
    categorias = list(set(fila[j] for fila in matrizDatos))
    print(f"Categorías del atributo {j}: {categorias}")

#b. Cuantitativo: mínimo, máximo, promedio, desviación estándar
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

#8. Elegir subconjunto de atributos a conservar (reducir columnas)
atributos_conservar = [int(x) for x in input("Ingresa los índices de atributos a conservar (separados por coma): ").split(",")]
matrizReducida = [[fila[i] for i in atributos_conservar] for fila in matrizDatos]
print("Matriz reducida por atributos:")
print(matrizReducida)

#9. Elegir subconjunto de renglones a conservar (reducir renglones)

print("¿Cómo deseas reducir renglones?")
print("1. Indicar renglones uno por uno")
print("2. Indicar un rango")
print("3. Filtrar por un valor en particular")
opcion = input("Opción: ")

def filtrarRenglones(matrizDatos):
    renglones_conservar = [int(x) for x in input("Ingresa los números de renglones a conservar (separados por coma): ").split(",")]
    matrizFiltrada = [matrizDatos[i] for i in renglones_conservar]
    return matrizFiltrada


def filtrarRango(matrizDatos):
    inicio = int(input("Index inicial: "))
    fin =int(input("Index final: "))
    matrizFiltrada = matrizDatos[inicio-1:fin+1] #+1 que incluye el indice final
    return matrizFiltrada
#5.9,3.0,5.1,1.8,Iris-virginica

def filtrarValor(matrizDatos):
    indice_columna = int(input("Índice de columna a filtrar: "))
    valor = input("Valor a filtrar: ")
    matrizFiltrada = [fila for fila in matrizDatos if fila[indice_columna] == valor]
    return matrizFiltrada

def filtrarMatriz(opcion, matrizDatos):
    if opcion == "1":
        return filtrarRenglones(matrizDatos)
    elif opcion == "2":
        return filtrarRango(matrizDatos)
    elif opcion == "3":
        return filtrarValor(matrizDatos)
    else:
        print("Opción no válida")
        return matrizDatos

matrizFinal = filtrarMatriz(opcion, matrizReducida)
print("Matriz reducida y filtrada:")
print(matrizFinal)

#10. Guardar el conjunto de datos filtrados en un archivo de texto plano
nombreSalida = input("Nombre del archivo de salida (sin extensión): ")
if not nombreSalida.endswith('.csv'):
    nombreSalida = nombreSalida + '.csv'
with open(nombreSalida, 'w', newline='', encoding='utf-8') as archivo:
    escribir = csv.writer(archivo, delimiter=separador)
    escribir.writerows(matrizFinal)
print(f"Archivo '{nombreSalida}' guardado con {len(matrizFinal)} renglones.")