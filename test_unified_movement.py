from collections import deque

def shortest_path(start, target):
    queue = deque([(start, [start])])
    visited = set()
    
    while queue:
        current_pos, path = queue.popleft()
        
        # Verificar si la posición actual está dentro del radio del objetivo
        if abs(current_pos[0] - target[0]) <= 1 and abs(current_pos[1] - target[1]) <= 1 and abs(current_pos[2] - target[2]) <= 1:
            path.append(target)  # Añadir la coordenada objetivo al camino
            return path
        
        if current_pos not in visited:
            visited.add(current_pos)
            
            # Calcular las diferencias entre las coordenadas actuales y el objetivo
            dx = target[0] - current_pos[0]
            dy = target[1] - current_pos[1]
            dz = target[2] - current_pos[2]
            
            # Determinar todas las posibles combinaciones de movimientos (incluyendo diagonales)
            possible_moves = []
            if dx != 0:
                possible_moves.append((dx // abs(dx), 0, 0))
            if dy != 0:
                possible_moves.append((0, dy // abs(dy), 0))
            if dz != 0:
                possible_moves.append((0, 0, dz // abs(dz)))
            if dx != 0 and dy != 0:
                possible_moves.append((dx // abs(dx), dy // abs(dy), 0))
            if dx != 0 and dz != 0:
                possible_moves.append((dx // abs(dx), 0, dz // abs(dz)))
            if dy != 0 and dz != 0:
                possible_moves.append((0, dy // abs(dy), dz // abs(dz)))
            
            # Agregar los próximos movimientos a la cola
            for move in possible_moves:
                next_move = (current_pos[0] + move[0], current_pos[1] + move[1], current_pos[2] + move[2])
                queue.append((next_move, path + [next_move]))


# Coordenadas especificadas inicialmente
specified_coordinates = [27, 15, 2]

# Coordenadas iniciales
start_coordinates = [10, 10, 2]

# Convertir las coordenadas especificadas a tupla
specified_coordinates_tuple = tuple(specified_coordinates)

# Convertir las coordenadas iniciales a tupla
start_coordinates_tuple = tuple(start_coordinates)

# Encontrar el camino más corto
shortest_path_coords = shortest_path(start_coordinates_tuple, specified_coordinates_tuple)
print("Camino más corto:", shortest_path_coords)
