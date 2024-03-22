import openai
import re
import argparse
from airsim_wrapper import *
import math
import numpy as np
import os
import json
import time
from openai import OpenAI

client = openai.OpenAI(api_key='API_KEY')

parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, default="prompts/airsim_basic.txt")
parser.add_argument("--sysprompt", type=str, default="system_prompts/airsim_basic.txt")
args = parser.parse_args()

with open("config.json", "r") as f:
    config = json.load(f)

print("Initializing ChatGPT...")
openai.api_key = config["OPENAI_API_KEY"]

with open(args.sysprompt, "r") as f:
    sysprompt = f.read()

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
        "role": "user",
        "content": "Avoid collisions with objects with the help of gpt-4-vision-preview "
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

    # Realizar detección de objetos antes de procesar la pregunta del usuario
    print("Realizando detección de objetos...")
    try:
        aw.perform_object_detection()  # Realizar detección de objetos en AirSim
        print("Detección de objetos completada.")
    except Exception as e:
        print(f"Error durante la detección de objetos: {e}")

    response = ask(question)

    print(f"\n{response}\n")

    code = extract_python_code(response)
    if code is not None:
        print("Please wait while I run the code in AirSim...")
        exec(extract_python_code(response))
        print("Done!\n")

    # Mover el dron a la siguiente posición después de cada interacción
    print("Moviendo el dron a la siguiente posición...")
    
    if "new_coords" in locals():
        aw.move_to_next_position(new_coords)  # Proporciona la posición objetivo a la función
        print("El dron se ha movido a la siguiente posición.")
    else:
        print("No se proporcionó una posición objetivo.")

