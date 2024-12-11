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
                        "text": "Dependiendo de lo que vez en la imagen hacia donde te moverias para intentar no chocar? -solo puedes responder con no moverse o si vez algo muy cerca responde con arriba, abajo, izquierda, derecha para que no choques"
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


'''
//////////////////////////////////////////////////
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



    # # Coordenadas especificadas inicialmente
    # specified_coordinates = ('AirSim> !move 5, 5, 2')
        
    # Coordenadas especificadas inicialmente
    specified_coordinates = [5, 5, 2]

    # aw.fly_to(specified_coordinates)
    
    # Coordenadas iniciales
    coordinates = [0, 0, 0]

    # # Convertir las coordenadas especificadas a cadenas
    # move_command = ('AirSim> !move', str(specified_coordinates[0]), str(specified_coordinates[1]), str(specified_coordinates[2]))

    # # Concatenar los elementos de la tupla en una cadena
    # move_command_str = ' '.join(str(item) for item in move_command)

    # # Imprimir la variable move_command_str
    # print(move_command_str)

    # Bucle while para iterar hasta que todas las coordenadas sean iguales o superiores a specified_coordinates
    while coordinates[0] < specified_coordinates[0] or coordinates[1] < specified_coordinates[1] or coordinates[2] < specified_coordinates[2]:
        
        if coordinates[0] != specified_coordinates[0]: # x
            coordinates[0] += 1

            aw.fly_to(coordinates)
            print('aw.fly_to(coordinates)')
            print(coordinates)
            response = visionTest()

            print('respuesta de vision 1', response['choices'][0]['message']['content'])
            
            time.sleep(15)
            
            # # Convertir las coordenadas especificadas a cadenas
            # move_command = ('AirSim> !move', str(coordinates[0]), str(coordinates[1]), str(coordinates[2]))
            # # Concatenar los elementos de la tupla en una cadena
            # move_command_str = ' '.join(str(item) for item in move_command)

            # response = ask(move_command_str)

            # print(f"\n{response}\n")

            # code = extract_python_code(response)
            # if code is not None:
            #     print("Please wait while I run the code in AirSim...")
            #     exec(extract_python_code(response))
            #     print("Done!\n")

        
        if coordinates[1] != specified_coordinates[1]: # y
            coordinates[1] += 1

            aw.fly_to(coordinates)
            print('aw.fly_to(coordinates)')
            print(coordinates)
            response = visionTest()
            
            print('respuesta de vision 2', response['choices'][0]['message']['content'])
            
            time.sleep(15)

            # # Convertir las coordenadas especificadas a cadenas
            # move_command = ('AirSim> !move', str(coordinates[0]), str(coordinates[1]), str(coordinates[2]))
            # # Concatenar los elementos de la tupla en una cadena
            # move_command_str = ' '.join(str(item) for item in move_command)

            # response = ask(move_command_str)

            # print(f"\n{response}\n")

            # code = extract_python_code(response)
            # if code is not None:
            #     print("Please wait while I run the code in AirSim...")
            #     exec(extract_python_code(response))
            #     print("Done!\n")


        if coordinates[2] != specified_coordinates[2]: # z
            coordinates[2] += 1

            aw.fly_to(coordinates)
            print('aw.fly_to(coordinates)')
            print(coordinates)
            response = visionTest()

            print('respuesta de vision 3', response['choices'][0]['message']['content'])

            time.sleep(15)

            # # Convertir las coordenadas especificadas a cadenas
            # move_command = ('AirSim> !move', str(coordinates[0]), str(coordinates[1]), str(coordinates[2]))
            # # Concatenar los elementos de la tupla en una cadena
            # move_command_str = ' '.join(str(item) for item in move_command)

            # response = ask(move_command_str)

            # print(f"\n{response}\n")

            # code = extract_python_code(response)
            # if code is not None:
            #     print("Please wait while I run the code in AirSim...")
            #     exec(extract_python_code(response))
            #     print("Done!\n")


        # Imprimir las coordenadas después de cada iteración
        print(coordinates)

    # Imprimir las coordenadas finales
    print("Las coordenadas finales son:", coordinates)




    # # # Utilizar las nuevas coordenadas en lugar de 'question' al llamar a la función ask
    # response = ask(move_command_str)

    # print(f"\n{response}\n")

    # code = extract_python_code(response)
    # if code is not None:
    #     print("Please wait while I run the code in AirSim...")
    #     exec(extract_python_code(response))
    #     print("Done!\n")
