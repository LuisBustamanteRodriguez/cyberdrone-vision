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
    start_time_chat = time.time()
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=chat_history,
        temperature=0
    )
    end_time_chat = time.time()
    print("Tiempo para solicitar chat:", end_time_chat - start_time_chat, "segundos")
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

    response = ask(question)

    print(f"\n{response}\n")

    code = extract_python_code(response)
    if code is not None:
        print("Please wait while I run the code in AirSim...")
        exec(extract_python_code(response))
        print("Done!\n")

    print("Realizando detección de objetos...")
    try:
        aw.perform_object_detection()
        print("Detección de objetos completada.")
    except Exception as e:
        print(f"Error durante la detección de objetos: {e}")

    print("Moviendo el dron a las coordenadas proporcionadas por el chatbot...")

    # Parsear las coordenadas del mensaje de respuesta
    match = re.search(r"\[(\d+),\s*(\d+),\s*(\d+)\]", response)
    if match:
        x, y, z = map(int, match.groups())
        new_coords = [x, y, z]
        start_time_movement = time.time()
        aw.fly_to(new_coords)
        end_time_movement = time.time()
        print("Tiempo de movimiento del dron:", end_time_movement - start_time_movement, "segundos")
        print("El dron se ha movido a las coordenadas proporcionadas por el chatbot.")
    else:
        print("No se proporcionaron coordenadas válidas.")