# Coordenadas especificadas inicialmente
specified_coordinates = [5, 5, 2]

# Coordenadas iniciales
coordinates = [0, 0, 0]

# Bucle while para iterar hasta que todas las coordenadas sean iguales o superiores a specified_coordinates
while coordinates[0] < specified_coordinates[0] or coordinates[1] < specified_coordinates[1] or coordinates[2] < specified_coordinates[2]:
    if coordinates[0] != specified_coordinates[0]:
        coordinates[0] += 1
    if coordinates[1] != specified_coordinates[1]:
        coordinates[1] += 1
    if coordinates[2] != specified_coordinates[2]:
        coordinates[2] += 1
    # Imprimir las coordenadas después de cada iteración
    print(coordinates)

# Imprimir las coordenadas finales
print("Las coordenadas finales son:", coordinates)
