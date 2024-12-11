import openai
import re
import argparse
import math
import numpy as np
import os
import json
import time
import airsim
from airsim_wrapper import *

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
    
    # # Capture image from AirSim
    # responses = airsim_client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)])
    # image_response = responses[0]
    # image_bytes = image_response.image_data_uint8
    # image = np.array(Image.frombytes("RGBA", (image_response.width, image_response.height), image_bytes))


    response = ask(question)

    print(f"\n{response}\n")

    code = extract_python_code(response)
    if code is not None:
        print("Please wait while I run the code in AirSim...")
        exec(extract_python_code(response))
        print("Done!\n")
