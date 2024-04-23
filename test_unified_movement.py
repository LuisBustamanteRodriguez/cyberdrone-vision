from collections import deque

def shortest_path(start, target):
    queue = deque([(start, [start])])
    visited = set()
    
    while queue:
        current_pos, path = queue.popleft()
        if current_pos == target:
            return path
        
        if current_pos not in visited:
            visited.add(current_pos)
            
            # Calcular las diferencias entre las coordenadas actuales y el objetivo
            dx = target[0] - current_pos[0]
            dy = target[1] - current_pos[1]
            dz = target[2] - current_pos[2]
            
            # Determinar hacia qué dirección movernos
            if abs(dx) >= max(abs(dy), abs(dz)):  # Movernos en la dirección X
                next_move = (current_pos[0] + dx // abs(dx), current_pos[1], current_pos[2])
            elif abs(dy) >= abs(dz):  # Movernos en la dirección Y
                next_move = (current_pos[0], current_pos[1] + dy // abs(dy), current_pos[2])
            else:  # Movernos en la dirección Z
                next_move = (current_pos[0], current_pos[1], current_pos[2] + dz // abs(dz))
            
            # Agregar el próximo movimiento a la cola
            queue.append((next_move, path + [next_move]))


# Coordenadas especificadas inicialmente
specified_coordinates = [-10, 15, 2]

# Coordenadas iniciales
start_coordinates = [0, 0, 2]

# Convertir las coordenadas especificadas a tupla
specified_coordinates_tuple = tuple(specified_coordinates)

# Convertir las coordenadas iniciales a tupla
start_coordinates_tuple = tuple(start_coordinates)

# Encontrar el camino más corto
shortest_path_coords = shortest_path(start_coordinates_tuple, specified_coordinates_tuple)
print("Camino más corto:", shortest_path_coords)
