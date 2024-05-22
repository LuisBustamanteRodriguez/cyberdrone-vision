import openai
import re
import argparse
import math
import numpy as np
import os
import json
import time
import base64
import requests
from PIL import Image
import airsim
import io
from airsim_wrapper import *
import json

import time
from collections import deque

from dotenv import load_dotenv

# Carga las variables de entorno desde el archivo .env
load_dotenv()

# Accede a la variable de entorno
api_key = os.getenv("OPENAI_API_KEY")

# Analizador de argumentos
parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, default="prompts/airsim_basic.txt")
parser.add_argument("--sysprompt", type=str, default="system_prompts/airsim_basic.txt")
args = parser.parse_args()

# Cargar configuración desde archivo JSON
with open("config.json", "r") as f:
    config = json.load(f)

# Imprimir mensaje de inicialización
print("Initializing ChatGPT...")
openai.api_key = config["OPENAI_API_KEY"]

# Leer contenido del archivo de sistema de prompt
with open(args.sysprompt, "r") as f:
    sysprompt = f.read()



# Establecer conexión con el cliente de AirSim una vez
conection = airsim.MultirotorClient()
conection.confirmConnection()



# =============== CHAT GPT ===================
    
# Historial de chat inicial
chat_history = [
    {
        "role": "system",
        "content": sysprompt
    },
    {
        "role": "user",
        "content": "move 10 units up"
    },
    {
        "role": "assistant",
        "content": """```python
        new_coords = [
            min(current_position[0], 30),
            min(current_position[1], 30),
            min(current_position[2] + 10, 30)
        ]
        aw.fly_to(new_coords)
        ```"""
    }
]

# Función para enviar una solicitud al modelo de lenguaje GPT-3.5
def ask(prompt):
    chat_history.append(
        {
            "role": "user",
            "content": prompt,
        }
    )
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=chat_history,
        temperature=0
    )
    chat_history.append(
        {
            "role": "assistant",
            "content": completion.choices[0].message.content,
        }
    )
    return chat_history[-1]["content"]


# =============== MOVEMENTS ===================

def move_drone(value, coordinates):
    if value == 'No moverse.':
        movement_north(coordinates)
    elif value == 'Izquierda':
        movement_west(coordinates)
    elif value == 'Derecha':
        movement_east(coordinates)
    elif value == 'Adelante':
        movement_south(coordinates)
    elif value == 'Arriba':
        movement_up(coordinates)
    elif value == 'Abajo':
        movement_down(coordinates)

'''///////////////////////////////////////////////////////////////////////////////////////////////////////////'''

def movement_north(coordinates):
    # get the current position of the drone
    current_yaw = aw.get_yaw()

    # set the position to current_yaw + 10 units (move forward 10 units)
    new_yaw = current_yaw + 0
    aw.set_yaw(new_yaw)
    print('girando hacia delante')
        
        
    coordinates[0] += 1

    aw.fly_to(coordinates)
    print("Nuevas coordenadas:", coordinates)
    

def movement_north_west(coordinates):
    # get the current yaw of the drone
    current_yaw = aw.get_yaw()

    # set the yaw to current_yaw - 45 degrees (turn 45 degrees to the left)
    new_yaw = current_yaw - 45
    aw.set_yaw(new_yaw)
    print('girando hacia el noroeste')

        
    coordinates[0] += 1
    coordinates[1] -= 1

    aw.fly_to(coordinates)
    print("Nuevas coordenadas:", coordinates)


def movement_north_east(coordinates):
    # get the current yaw of the drone
    current_yaw = aw.get_yaw()

    # set the yaw to current_yaw - 45 degrees (turn 45 degrees to the left)
    new_yaw = current_yaw + 45
    aw.set_yaw(new_yaw)
    print('girando hacia el noreste')

        
    coordinates[0] += 1
    coordinates[1] += 1

    aw.fly_to(coordinates)
    print("Nuevas coordenadas:", coordinates)


def movement_west(coordinates):
    # get the current yaw of the drone
    current_yaw = aw.get_yaw()

    # set the yaw to current_yaw - 90 degrees (turn 90 degrees to the left)
    new_yaw = current_yaw - 90
    aw.set_yaw(new_yaw)
    print('girando hacia la izquierda')

        
    coordinates[1] -= 1

    aw.fly_to(coordinates)
    print("Nuevas coordenadas:", coordinates)


def movement_east(coordinates):
    # get the current yaw of the drone
    current_yaw = aw.get_yaw()

    # set the yaw to current_yaw + 90 degrees (turn 90 degrees to the right)
    new_yaw = current_yaw + 90
    aw.set_yaw(new_yaw)
    print('girando hacia la derecha')
        

    coordinates[1] += 1

    aw.fly_to(coordinates)
    print("Nuevas coordenadas:", coordinates)


def movement_south(coordinates):
    # get the current position of the drone
    current_yaw = aw.get_yaw()

    # set the position to current_position - 10 units (move backward 10 units)
    new_yaw = current_yaw + 180
    aw.set_yaw(new_yaw)
    print('girando hacia atras')


    coordinates[0] -= 1

    aw.fly_to(coordinates)
    print("Nuevas coordenadas:", coordinates)


def movement_south_west(coordinates):
    # get the current position of the drone
    current_yaw = aw.get_yaw()

    new_yaw = current_yaw - 135
    aw.set_yaw(new_yaw)
    print('girando hacia el suroeste')


    coordinates[0] -= 1
    coordinates[1] -= 1

    aw.fly_to(coordinates)
    print("Nuevas coordenadas:", coordinates)


def movement_south_east(coordinates):

    # get the current position of the drone
    current_yaw = aw.get_yaw()

    new_yaw = current_yaw + 135
    aw.set_yaw(new_yaw)
    print('girando hacia el sureste')

    coordinates[0] -= 1
    coordinates[1] += 1

    aw.fly_to(coordinates)
    print("Nuevas coordenadas:", coordinates)


def movement_up(coordinates):
    movement = False
        
    coordinates[2] += 3

    aw.fly_to(coordinates)
    print("Nuevas coordenadas:", coordinates)
    print('moviendo hacia arriba')


def movement_down(coordinates):
    start_time = time.time()
    movement = False

    coordinates[2] -= 1

    aw.fly_to(coordinates)
    print("Nuevas coordenadas:", coordinates)
    print('moviendo hacia abajo')


'''
/////////////////////////////////////////////////////////////
'''

# =============== shortest_path ===================

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



'''
/////////////////////////////////////////////////////////////
'''

print(f"Done.")

code_block_regex = re.compile(r"```(.*?)```", re.DOTALL)

def extract_python_code(content):
    code_blocks = code_block_regex.findall(content)
    if code_blocks:
        full_code = "\n".join(code_blocks)

        if full_code.startswith("python"):
            full_code = full_code[7:]

        return full_code
    else:
        return None


print(f"Initializing AirSim...")
aw = AirSimWrapper()
print(f"Done.")

with open(args.prompt, "r") as f:
    prompt = f.read()

ask(prompt)
print("Welcome to the AirSim chatbot! I am ready to help you with your AirSim questions and commands.")

while True:
    question = input("AirSim> ")
    
    if question == "!quit" or question == "!exit":
        break


    if question == "!clear":
        os.system("cls")
        continue
    
    response = ask(question)
    
    visionTest()

    print(f"\n{response}\n")

    code = extract_python_code(response)
    if code is not None:
        print("Please wait while I run the code in AirSim...")
        exec(extract_python_code(response))
        print("Done!\n")


        
    # Coordenadas especificadas inicialmente
    specified_coordinates = [27, 15, 2]

    breackpoint1 = [15, 10, 2]

    # calculate coordenates
    coordinates_math = specified_coordinates

    # Coordenadas iniciales
    coordinates = [0, 0, 2]

    # Convertir las coordenadas especificadas a tupla
    specified_coordinates_tuple = tuple(specified_coordinates)

    breakpoint1_tuple = tuple(breackpoint1)

    # Convertir las coordenadas iniciales a tupla
    start_coordinates_tuple = tuple(coordinates)

    # Encontrar el camino más corto
    shortest_path_coords_breackpoint = shortest_path(start_coordinates_tuple, breakpoint1_tuple)
    print("Camino más breackpoint 1:", shortest_path_coords_breackpoint)

    shortest_path_coords = shortest_path(breakpoint1_tuple, specified_coordinates_tuple)
    print("Camino más corto:", shortest_path_coords)

    old_coordinates = [0, 0, 2]

    # Bucle while para iterar hasta que todas las coordenadas sean iguales o superiores a specified_coordinates
    while coordinates[0] <= specified_coordinates[0] or coordinates[1] <= specified_coordinates[1] or coordinates[2] <= specified_coordinates[2]:


        for location in shortest_path_coords_breackpoint:
            print("Ubicación:", location)
            # aw.fly_to(location)

            new_coordinates = [location[0], location[1], location[2]]
            # print("Nuevas coordenadas:", new_coordinates).
            movement_coordinates = new_coordinates

            old_coordinates_temp = old_coordinates[:]

            print('////////////////')
            print("Nuevas coordenadas:", new_coordinates[0])
            print("Nuevas coordenadas:", new_coordinates[1])
            print("Nuevas coordenadas:", new_coordinates[2])
            
            print('old coordinates:', old_coordinates[0])
            print('old coordinates:', old_coordinates[1])
            print('old coordinates:', old_coordinates[2])


            if new_coordinates[0] > old_coordinates[0]:
                if new_coordinates[1] > old_coordinates[1]:
                    movement_north_east(old_coordinates_temp)

                elif new_coordinates[1] < old_coordinates[1]:
                    movement_north_west(old_coordinates_temp)
                else:
                    movement_north(old_coordinates_temp)

            
            elif new_coordinates[0] < old_coordinates[0]:
                if new_coordinates[1] > old_coordinates[1]:
                    movement_south_east(old_coordinates_temp)
                    
                elif new_coordinates[1] < old_coordinates[1]:
                    movement_south_west(old_coordinates_temp)     
                else:
                    movement_south(old_coordinates_temp)
                    
            elif new_coordinates[0] == old_coordinates[0]:
                if new_coordinates[1] > old_coordinates[1]:
                    movement_east(old_coordinates_temp)
                    
                elif new_coordinates[1] < old_coordinates[1]:
                    movement_west(old_coordinates_temp)
                    

            # Movimiento en Z
            if new_coordinates[2] > old_coordinates[2]:
                print("Movimiento hacia arriba")
                movement_up(old_coordinates_temp)
            elif new_coordinates[2] < old_coordinates[2]:
                print("Movimiento hacia abajo")
                movement_down(old_coordinates_temp)
        

            # Actualizar old_coordinates al final de la iteración
            old_coordinates = new_coordinates[:]


        # Recorrer todas las ubicaciones en el camino más corto
        for location in shortest_path_coords:
            print("Ubicación:", location)
            # aw.fly_to(location)
        
            new_coordinates = [location[0], location[1], location[2]]
            # print("Nuevas coordenadas:", new_coordinates).
            movement_coordinates = new_coordinates

            old_coordinates_temp = old_coordinates[:]

            print('////////////////')
            print("Nuevas coordenadas:", new_coordinates[0])
            print("Nuevas coordenadas:", new_coordinates[1])
            print("Nuevas coordenadas:", new_coordinates[2])
            
            print('old coordinates:', old_coordinates[0])
            print('old coordinates:', old_coordinates[1])
            print('old coordinates:', old_coordinates[2])

            
            if new_coordinates[0] > old_coordinates[0]:
                if new_coordinates[1] > old_coordinates[1]:
                    movement_north_east(old_coordinates_temp)

                elif new_coordinates[1] < old_coordinates[1]:
                    movement_north_west(old_coordinates_temp)
                else:
                    movement_north(old_coordinates_temp)

        
            elif new_coordinates[0] < old_coordinates[0]:
                if new_coordinates[1] > old_coordinates[1]:
                    movement_south_east(old_coordinates_temp)
                    
                elif new_coordinates[1] < old_coordinates[1]:
                    movement_south_west(old_coordinates_temp)     
                else:
                    movement_south(old_coordinates_temp)
                    
            elif new_coordinates[0] == old_coordinates[0]:
                if new_coordinates[1] > old_coordinates[1]:
                    movement_east(old_coordinates_temp)
                    
                elif new_coordinates[1] < old_coordinates[1]:
                    movement_west(old_coordinates_temp)
                

            # Movimiento en Z
            if new_coordinates[2] > old_coordinates[2]:
                print("Movimiento hacia arriba")
                movement_up(old_coordinates_temp)
            elif new_coordinates[2] < old_coordinates[2]:
                print("Movimiento hacia abajo")
                movement_down(old_coordinates_temp)
            

            # Actualizar old_coordinates al final de la iteración
            old_coordinates = new_coordinates[:]



        coordinates = new_coordinates[:]
        print("coordenadas finales:", coordinates)
        print('finalizo el recorrido')
            


