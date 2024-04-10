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

# Clave de la API de OpenAI
client = openai.OpenAI(api_key='api_key')

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


   # =============== VISION ===================
    
# Función para capturar una imagen desde AirSim
def capture_image_from_airsim():
    # Conectar con el cliente de AirSim
    client = airsim.MultirotorClient()
    client.confirmConnection()

    # Capturar una imagen utilizando la cámara frontal
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])

    # Obtener la imagen de la respuesta
    image_response = responses[0]
    image_bytes = image_response.image_data_uint8
    
    # Convertir los bytes de la imagen a un objeto Image de Pillow
    image = Image.frombytes("RGB", (image_response.width, image_response.height), image_bytes)
    
    return image

# Función para convertir la imagen de AirSim a un formato compatible con Vision
def convert_image_for_vision(image):
    image_vision_format = image.convert("RGB")  # RGB es el formato que usa AirSim
    
    return image_vision_format

# Función para enviar una solicitud a la API de OpenAI para pruebas de visión
def visionTest():
    image_from_airsim = capture_image_from_airsim()

    image_vision_format = convert_image_for_vision(image_from_airsim)

    # Obtener la imagen en base64 desde la imagen convertida png
    buffer = io.BytesIO()
    image_vision_format.save(buffer, format="PNG")
    base64_image = base64.b64encode(buffer.getvalue()).decode()

    # Configuración de los headers para la solicitud a la API de OpenAI
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {client.api_key}"
    }

    # Configuración del payload para la solicitud a la API de OpenAI
    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "tu concentrate en el centro de la imagen si no vez nada delante de ti en el centro responde con no moverse. Si vez algo enfrente tuyo responde hacia que lado es mejor girar -solo puedes responder con arriba, abajo, izquierda, derecha para que no choques"
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 50
    }

    # Realizar la solicitud a la API de OpenAI
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    print(response.json())

    return response.json()


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
        movement_forward(coordinates)
    elif value == 'Izquierda':
        movement_left(coordinates)
    elif value == 'Derecha':
        movement_right(coordinates)
    elif value == 'Adelante':
        movement_backward(coordinates)
    elif value == 'Arriba':
        movement_up(coordinates)
    elif value == 'Abajo':
        movement_down(coordinates)

'''///////////////////////////////////////////////////////////////////////////////////////////////////////////'''

def movement_left(coordinates):
    start_time = time.time()
    movement = False
    # get the current yaw of the drone
    current_yaw = aw.get_yaw()

    # set the yaw to current_yaw - 90 degrees (turn 90 degrees to the left)
    new_yaw = current_yaw - 90
    aw.set_yaw(new_yaw)
    print('girando hacia la izquierda')

    while movement == False:

        response = visionTest()
        print('respuesta de vision 1', response['choices'][0]['message']['content'])
        value = response['choices'][0]['message']['content']
        # calcular tiempo de ejecución
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Tiempo de ejecución de movement_left: {execution_time} segundos")

        if value == 'Izquierda':
            movement_forward(coordinates)
        elif value == 'Derecha':
            movement_backward(coordinates)
        elif value == 'Arriba':
            movement_up(coordinates)
        elif value == 'Abajo':
            movement_down(coordinates)
        
        coordinates[1] -= 3

        aw.fly_to(coordinates)
        print("Nuevas coordenadas:", coordinates)


def movement_right(coordinates):
    start_time = time.time()
    movement = False
    # get the current yaw of the drone
    current_yaw = aw.get_yaw()

    # set the yaw to current_yaw + 90 degrees (turn 90 degrees to the right)
    new_yaw = current_yaw + 90
    aw.set_yaw(new_yaw)
    print('girando hacia la derecha')

    while movement == False:

        response = visionTest()
        print('respuesta de vision 1', response['choices'][0]['message']['content'])
        value = response['choices'][0]['message']['content']
        # calcular tiempo de ejecución
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Tiempo de ejecución de movement_right: {execution_time} segundos")

        
        if value == 'Izquierda':
            movement_forward(coordinates)
        elif value == 'Derecha':
            movement_backward(coordinates)
        elif value == 'Arriba':
            movement_up(coordinates)
        elif value == 'Abajo':
            movement_down(coordinates)
        
        coordinates[1] += 3

        aw.fly_to(coordinates)

        print("Nuevas coordenadas:", coordinates)


       
        


def movement_forward(coordinates):
    start_time = time.time()
    movement = False
    # get the current position of the drone
    current_yaw = aw.get_yaw()

    # set the position to current_yaw + 10 units (move forward 10 units)
    new_yaw = current_yaw + 0
    aw.set_yaw(new_yaw)
    print('girando hacia delante')

    while movement == False:

        response = visionTest()
        print('respuesta de vision 1', response['choices'][0]['message']['content'])
        value = response['choices'][0]['message']['content']
        # calcular tiempo de ejecución
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Tiempo de ejecución de movement_forward: {execution_time} segundos")
        

        if value == 'Izquierda':
            movement_left(coordinates)
        elif value == 'Derecha':
            movement_right(coordinates)
        elif value == 'Arriba':
            movement_up(coordinates)
        elif value == 'Abajo':
            movement_down(coordinates)
        
        coordinates[0] += 3

        aw.fly_to(coordinates)

        print("Nuevas coordenadas:", coordinates)
    
def movement_backward(coordinates):
    start_time = time.time()
    movement = False
    # get the current position of the drone
    current_yaw = aw.get_yaw()

    # set the position to current_position - 10 units (move backward 10 units)
    new_yaw = current_yaw + 180
    aw.set_yaw(new_yaw)
    print('girando hacia atras')

    while movement == False:

        response = visionTest()
        print('respuesta de vision 1', response['choices'][0]['message']['content'])
        value = response['choices'][0]['message']['content']
        # calcular tiempo de ejecución
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Tiempo de ejecución de movement_backward: {execution_time} segundos")


        if value == 'Izquierda':
            movement_right(coordinates)
        elif value == 'Derecha':
            movement_left(coordinates)
        elif value == 'Arriba':
            movement_up(coordinates)
        elif value == 'Abajo':
            movement_down(coordinates)

        coordinates[0] -= 3

        aw.fly_to(coordinates)
        print("Nuevas coordenadas:", coordinates)
        
def movement_up(coordinates):
    start_time = time.time()
    movement = False
        
    coordinates[2] += 3

    aw.fly_to(coordinates)
    print("Nuevas coordenadas:", coordinates)
    print('moviendo hacia arriba')

    response = visionTest()
    print('respuesta de vision 1', response['choices'][0]['message']['content'])
    value = response['choices'][0]['message']['content']
    # calcular tiempo de ejecución
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Tiempo de ejecución de movement_up: {execution_time} segundos")

    if value == 'No moverse.':
        movement_forward(coordinates)
    elif value == 'Izquierda':
        movement_left(coordinates)
    elif value == 'Derecha':
        movement_right(coordinates)
    elif value == 'Arriba':
        movement_up(coordinates)
    elif value == 'Abajo':
        movement_down(coordinates)

def movement_down(coordinates):
    start_time = time.time()
    movement = False

    coordinates[2] -= 3

    aw.fly_to(coordinates)
    print("Nuevas coordenadas:", coordinates)
    print('moviendo hacia abajo')

    response = visionTest()
    print('respuesta de vision 1', response['choices'][0]['message']['content'])
    value = response['choices'][0]['message']['content']
    # calcular tiempo de ejecución
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Tiempo de ejecución de movement_down: {execution_time} segundos")

    if value == 'No moverse.':
        movement_forward(coordinates)
    elif value == 'Izquierda':
        movement_left(coordinates)
    elif value == 'Derecha':
        movement_right(coordinates)
    elif value == 'Arriba':
        movement_up(coordinates)
    elif value == 'Abajo':
        movement_down(coordinates)

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
    specified_coordinates = [5, 5, 2]
    
    # Coordenadas iniciales
    coordinates = [0, 0, 2]

    # Bucle while para iterar hasta que todas las coordenadas sean iguales o superiores a specified_coordinates
    while coordinates[0] < specified_coordinates[0] or coordinates[1] < specified_coordinates[1] or coordinates[2] < specified_coordinates[2]:
        

        response = visionTest()
        print('respuesta de vision 1', response['choices'][0]['message']['content'])
        value = response['choices'][0]['message']['content']
        
        if value == 'No moverse.':
            movement_forward(coordinates)

        elif value == 'Izquierda':
            movement_left(coordinates)

        elif value == 'Derecha':
             movement_right(coordinates)

        elif value == 'Arriba':
            movement_up(coordinates)
        
        elif value == 'Abajo':
            movement_down(coordinates)

        # Imprimir las coordenadas después de cada iteración
        print(coordinates)

    # Imprimir las coordenadas finales
    print("Las coordenadas finales son:", coordinates)
