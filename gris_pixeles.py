from PIL import Image

def contar_colores(imagen):
    # Abrir la imagen
    img = Image.open(imagen).convert('I')  # 'I' mode es para imágenes de 32 bits, pero 'L' solo permite 8 bits
    
    # Obtener las dimensiones de la imagen
    ancho, alto = img.size
    
    # Inicializar los contadores
    negros = 0
    blancos = 0
    grises = 0
    
    # Recorrer cada píxel de la imagen
    for y in range(alto):
        for x in range(ancho):
            pixel = img.getpixel((x, y))
            if pixel == 0:
                negros += 1
            elif pixel == 65535:
                blancos += 1
            else:
                grises += 1
    
    return negros, blancos, grises

# Ejemplo de uso
imagen = "D:/Dron/cyberdrone-vision/depth_16bit.png"
negros, blancos, grises = contar_colores(imagen)

print(f"Negros: {negros}")
print(f"Blancos: {blancos}")
print(f"Grises: {grises}")

