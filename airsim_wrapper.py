import airsim
import numpy as np
import cv2
from ultralytics import YOLO
import time

class ObjectDetector:
    def __init__(self):
        self.model = YOLO("yolov8l.pt")
        self.model.conf = 0.5

    def detect_objects(self, image):
        results = self.model(image)
        return results

class AirSimWrapper:
    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.object_detector = ObjectDetector()
        self.last_darkest_pixels = None  # Guardar referencia de los píxeles más oscuros
        self.has_taken_off = False  # Bandera para controlar si el dron ha despegado

    def takeoff(self):
        self.client.takeoffAsync().join()
        self.has_taken_off = True  # Marcar que el dron ha despegado

    def land(self):
        self.client.landAsync().join()
        self.has_taken_off = False  # Reiniciar la bandera cuando el dron aterrice

    def get_drone_position(self):
        pose = self.client.simGetVehiclePose()
        return [pose.position.x_val, pose.position.y_val, pose.position.z_val]

    def fly_to(self, point):
        if self.has_taken_off:  # Verificar si el dron ha despegado
            # Mover el dron a la posición objetivo
            self.client.moveToPositionAsync(point[0], point[1], point[2], 5).join()
        else:
            # Si el dron aún no ha despegado, simplemente despegarlo
            self.takeoff()


    def fly_path(self, points):
        airsim_points = []
        for point in points:
            if point[2] > 0:
                airsim_points.append(airsim.Vector3r(point[0], point[1], -point[2]))
            else:
                airsim_points.append(airsim.Vector3r(point[0], point[1], point[2]))
        self.client.moveOnPathAsync(airsim_points, 5, 120, airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False, 0), 20, 1).join()

    def set_yaw(self, yaw):
        self.client.rotateToYawAsync(yaw, 5).join()

    def get_yaw(self):
        orientation_quat = self.client.simGetVehiclePose().orientation
        yaw = airsim.to_eularian_angles(orientation_quat)[2]
        return yaw

    # En el método get_drone_position(), devuelve las coordenadas x e y como una lista o tupla
    def get_drone_position(self):
        pose = self.client.simGetVehiclePose()
        return pose.position.x_val, pose.position.y_val

    def process_image(self, image):
        # Convertir la imagen a escala de grises
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Encontrar los píxeles más oscuros
        darkest_pixel_value = np.min(gray_image)
        darkest_pixels = np.where(gray_image == darkest_pixel_value)
        
        # Convertir coordenadas de píxeles a formato adecuado
        darkest_pixels_position = np.column_stack((darkest_pixels[0], darkest_pixels[1]))
        
        # Obtener la posición del dron
        drone_position = np.array(self.get_drone_position())  # Obtener las coordenadas x e y del dron
        
        # Obtener la posición relativa de los píxeles más oscuros respecto al dron
        relative_position = darkest_pixels_position - drone_position

        # Tomar acciones para evitar colisiones basadas en la posición relativa de los píxeles más oscuros
        return self.avoid_collision(relative_position)

    def avoid_collision(self, relative_position):
        # Define la distancia máxima que el dron se moverá para evitar la colisión
        max_distance_to_move = 5

        # Obtener la magnitud de la posición relativa
        distance_to_object = np.linalg.norm(relative_position, axis=1)

        # Determinar si el objeto está lo suficientemente cerca como para requerir movimiento
        if np.any(distance_to_object < max_distance_to_move):
            # Obtener la dirección relativa de los píxeles más oscuros
            direction_to_move = np.mean(relative_position, axis=0)

            # Normalizar la dirección y multiplicar por la distancia máxima
            direction_to_move /= np.linalg.norm(direction_to_move)
            direction_to_move *= max_distance_to_move

            # Convertir la dirección al sistema de coordenadas del dron
            dx, dy = direction_to_move

            # Mover el dron en la dirección calculada
            print(f"Moviendo el dron {max_distance_to_move} metros en la dirección {direction_to_move}")
            self.client.moveByVelocityAsync(dx, dy, 0, 1).join()

            # Esperar un momento antes de volver a realizar la detección de objetos
            time.sleep(1)
            self.perform_object_detection()  # Realizar la detección de objetos nuevamente

            return True

        # Si no se detectan objetos lo suficientemente cerca, indicar que no hay riesgo de colisión
        return False


    
    def get_relative_position_of_darkest_pixels(self):
        # Realizar la detección de objetos para obtener la nueva posición relativa de los píxeles más oscuros
        responses = self.client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.Scene, False, False)])
        response = responses[0]
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(response.height, response.width, 3)

        image = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

        # Convertir la imagen a escala de grises
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Encontrar los píxeles más oscuros
        darkest_pixel_value = np.min(gray_image)
        darkest_pixels = np.where(gray_image == darkest_pixel_value)
        
        # Convertir coordenadas de píxeles a formato adecuado
        darkest_pixels_position = np.column_stack((darkest_pixels[0], darkest_pixels[1]))

        # Obtener la posición del dron
        drone_position = np.array(self.get_drone_position()[:2])  # Obtener las coordenadas x e y del dron

        # Obtener la posición relativa de los píxeles más oscuros respecto al dron
        relative_position = darkest_pixels_position - drone_position

        return relative_position


    def perform_object_detection(self):
        # Realizar la detección de objetos solo si el dron ha despegado
        if self.has_taken_off:
            responses = self.client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.Scene, False, False)])
            response = responses[0]
            img1d = np.frombuffer(response.image_data_uint8, dtype=np.uint8)
            img_rgb = img1d.reshape(response.height, response.width, 3)

            image = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
            result = self.process_image(image)


if __name__ == "__main__":
    airsim_wrapper = AirSimWrapper()
    airsim_wrapper.takeoff()
    airsim_wrapper.perform_object_detection()
    airsim_wrapper.land()
