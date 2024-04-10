import openai
import re
import argparse
import math
import numpy as np
import os
import json
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


   # =============== EXTRACT IMAGE ===================
    
# Función para capturar una imagen desde AirSim
def capture_image_from_airsim():
    # Conectar con el cliente de AirSim
    client = airsim.MultirotorClient()
    client.confirmConnection()

    # Capturar una imagen utilizando la cámara frontal
    responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])

    # get camera images from the car
    images = client.simGetImages([
        airsim.ImageRequest("0", airsim.ImageType.DepthVis),  #depth visualization image
        airsim.ImageRequest("1", airsim.ImageType.DepthPerspective, True), #depth in perspective projection
        airsim.ImageRequest("1", airsim.ImageType.Scene), #scene vision image in png format
        airsim.ImageRequest("1", airsim.ImageType.Scene, False, False)])  #scene vision image in uncompressed RGBA array
    print('Retrieved images: %d' % len(images))


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



    # =============== VISION ===================

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
                        "text": "como vez la imagen?"
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


print(f"Initializing AirSim Vision Sends...")
aw = AirSimWrapper()
print(f"Done.")

with open(args.prompt, "r") as f:
    prompt = f.read()

print("Welcome to the AirSim chatbot! I am ready to help you with your AirSim questions and commands.")


while True: 

    response = visionTest()
    print('respuesta de vision 1', response['choices'][0]['message']['content'])
    value = response['choices'][0]['message']['content']
        
       