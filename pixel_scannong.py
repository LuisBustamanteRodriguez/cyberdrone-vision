# from PIL import Image

# def obtener_valores_de_pixeles(imagen):
#     # Abrir la imagen
#     img = Image.open(imagen)
    
#     # Obtener las dimensiones de la imagen
#     ancho, alto = img.size
    
#     # Crear una matriz para almacenar los valores de los píxeles
#     matriz_pixeles = []
    
#     # Recorrer cada píxel de la imagen
#     for y in range(alto):
#         fila = []
#         for x in range(ancho):
#             # Obtener el valor RGB del píxel en la posición (x, y)
#             pixel = img.getpixel((x, y))
#             fila.append(pixel)
#         matriz_pixeles.append(fila)
    
#     return matriz_pixeles

# # Ejemplo de uso
# imagen = "D:\Dron\cyberdrone-vision\depth_16bit.png"  
# matriz_valores = obtener_valores_de_pixeles(imagen)
# print(matriz_valores)



# from PIL import Image

# def contar_pixeles_blancos(imagen):
#     # Abrir la imagen
#     img = Image.open(imagen)
    
#     # Obtener las dimensiones de la imagen
#     ancho, alto = img.size
    
#     # Definir las coordenadas de las secciones
#     izquierda = (0, 0, ancho//3, alto)
#     centro = (ancho//3, 0, 2*(ancho//3), alto)
#     derecha = (2*(ancho//3), 0, ancho, alto)
    
#     # Contar píxeles blancos en cada sección
#     blancos_izquierda = sum(1 for pixel in img.crop(izquierda).getdata() if pixel == (255, 255, 255))
#     blancos_centro = sum(1 for pixel in img.crop(centro).getdata() if pixel == (255, 255, 255))
#     blancos_derecha = sum(1 for pixel in img.crop(derecha).getdata() if pixel == (255, 255, 255))
    
#     return blancos_izquierda, blancos_centro, blancos_derecha

# # Ejemplo de uso
# imagen = "D:/Dron/cyberdrone-vision/depth_16bit.png"  
# blancos_izquierda, blancos_centro, blancos_derecha = contar_pixeles_blancos(imagen)

# if blancos_izquierda > blancos_centro and blancos_izquierda > blancos_derecha:
#     print("Hay más píxeles blancos en el lado izquierdo.")
# elif blancos_centro > blancos_izquierda and blancos_centro > blancos_derecha:
#     print("Hay más píxeles blancos en el centro.")
# else:
#     print("Hay más píxeles blancos en el lado derecho.")



from PIL import Image
import matplotlib.pyplot as plt

def contar_pixeles_blancos(imagen):
    # Abrir la imagen
    img = Image.open(imagen)
    
    # Obtener las dimensiones de la imagen
    ancho, alto = img.size
    
    # Definir las coordenadas de las secciones
    izquierda = (0, 0, ancho//3, alto)
    centro = (ancho//3, 0, 2*(ancho//3), alto)
    derecha = (2*(ancho//3), 0, ancho, alto)
    
    # Contar píxeles blancos en cada sección
    blancos_izquierda = sum(1 for pixel in img.crop(izquierda).getdata() if pixel == 65535)
    blancos_centro = sum(1 for pixel in img.crop(centro).getdata() if pixel == 65535)
    blancos_derecha = sum(1 for pixel in img.crop(derecha).getdata() if pixel == 65535)
    
    return blancos_izquierda, blancos_centro, blancos_derecha

# Ejemplo de uso
imagen = "D:/Dron/cyberdrone-vision/depth_16bit.png"
blancos_izquierda, blancos_centro, blancos_derecha = contar_pixeles_blancos(imagen)

# Datos para el gráfico
secciones = ['Izquierda', 'Centro', 'Derecha']
valores = [blancos_izquierda, blancos_centro, blancos_derecha]

# Crear el gráfico de barras
plt.bar(secciones, valores, color=['blue', 'green', 'red'])
plt.xlabel('Sección de la imagen')
plt.ylabel('Número de píxeles blancos')
plt.title('Distribución de píxeles blancos en la imagen')
plt.show()

# Mostrar los resultados en texto también
if blancos_izquierda > blancos_centro and blancos_izquierda > blancos_derecha:
    print("Hay más píxeles blancos en el lado izquierdo.")
elif blancos_centro > blancos_izquierda and blancos_centro > blancos_derecha:
    print("Hay más píxeles blancos en el centro.")
else:
    print("Hay más píxeles blancos en el lado derecho.")