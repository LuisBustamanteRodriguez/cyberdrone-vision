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
        self.last_darkest_pixels = None  
        self.has_taken_off = False  
        self.distance_threshold = 1  
        self.last_detection_position = None


    def takeoff(self):
        start_time_takeoff = time.time()
        self.client.takeoffAsync().join()
        end_time_takeoff = time.time()
        print("Tiempo de despegue:", end_time_takeoff - start_time_takeoff, "segundos")
        self.has_taken_off = True  # Marcar que el dron ha despegado

    def land(self):
        start_time_land = time.time()
        self.client.landAsync().join()
        end_time_land = time.time()
        print("Tiempo de aterrizaje:", end_time_land - start_time_land, "segundos")
        self.has_taken_off = False  # Reiniciar la bandera cuando el dron aterrice

    def get_drone_position(self):
        pose = self.client.simGetVehiclePose()
        return [pose.position.x_val, pose.position.y_val, pose.position.z_val]

    def fly_to(self, point):
        if isinstance(point, str) and point.lower() == 'take off':
            # Si se indica despegue, despegar el dron y marcar que ha despegado
            self.takeoff()
            return

        # Si el dron aún no ha despegado, simplemente moverlo a la posición objetivo sin realizar la detección de objetos
        if not self.has_taken_off:
            if point[2] > 0:
                target_position = [point[0], point[1], -point[2]] 
            else:
                target_position = point
            self.client.moveToPositionAsync(target_position[0], target_position[1], target_position[2], 5).join()
            return

        # Si el dron ha despegado, capturar la imagen de la cámara
        responses = self.client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.Scene, False, False)])
        response = responses[0]
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(response.height, response.width, 3)
        image = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

        
        collision_avoided = self.process_image(image)

        
        if point[2] > 0:
            target_position = [point[0], point[1], -point[2]]
        else:
            target_position = point

        # Mover el dron a la posición objetivo
        self.client.moveToPositionAsync(target_position[0], target_position[1], target_position[2], 5).join()

    def fly_path(self, points):
        airsim_points = []
        for point in points:
            if point[2] > 0:
                airsim_points.append(airsim.Vector3r(point[0], point[1], -point[2]))
            else:
                airsim_points.append(airsim.Vector3r(point[0], point[1], point[2]))
        start_time_fly_path = time.time()
        self.client.moveOnPathAsync(airsim_points, 5, 120, airsim.DrivetrainType.ForwardOnly, airsim.YawMode(False, 0), 20, 1).join()
        end_time_fly_path = time.time()
        print("Tiempo de vuelo del trayecto:", end_time_fly_path - start_time_fly_path, "segundos")

    def set_yaw(self, yaw):
        start_time_set_yaw = time.time()
        self.client.rotateToYawAsync(yaw, 5).join()
        end_time_set_yaw = time.time()
        print("Tiempo de ajuste del ángulo de giro:", end_time_set_yaw - start_time_set_yaw, "segundos")

    def get_yaw(self):
        orientation_quat = self.client.simGetVehiclePose().orientation
        yaw = airsim.to_eularian_angles(orientation_quat)[2]
        return yaw

    
    def get_drone_position(self):
        pose = self.client.simGetVehiclePose()
        return pose.position.x_val, pose.position.y_val

    def process_image(self, image):
        # Convertir la imagen a escala de grises
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Mapear el color de los píxeles a una aproximación de distancia
        distance_map = 255 - gray_image

        # Encontrar el píxel más oscuro
        darkest_pixel_value = np.min(distance_map)
        darkest_pixels = np.where(distance_map == darkest_pixel_value)
        
        # Tomar solo el primer píxel más oscuro
        darkest_pixel_position = np.array([darkest_pixels[0][0], darkest_pixels[1][0]])
        
        # Obtener la posición del dron
        drone_position = np.array(self.get_drone_position())  # Convertir las coordenadas x, y, z del dron en un array NumPy
        
        # Obtener la posición relativa del píxel más oscuro respecto al dron
        relative_position = darkest_pixel_position - drone_position

        # Tomar acciones para evitar colisiones basadas en la posición relativa del píxel más oscuro
        return self.avoid_collision(relative_position)

    def avoid_collision(self, relative_position, distance_threshold=100):
        distances = np.linalg.norm(relative_position, axis=1)

        close_pixels_indices = np.where(distances < distance_threshold)[0]

        if len(close_pixels_indices) > 0:
            print("Se detectaron objetos cercanos. Tomando medidas evasivas...")

            # Hay píxeles oscuros en la parte superior de la imagen
            if np.any(relative_position[:, 1] < 0):
                print("Píxeles oscuros detectados arriba, moviéndose hacia abajo...")
                start_time_move_down = time.time()
                self.client.moveByVelocityAsync(0, 0, 1, 1).join()
                end_time_move_down = time.time()
                print("Tiempo de movimiento hacia abajo:", end_time_move_down - start_time_move_down, "segundos")
                time.sleep(1)
                start_time_detection = time.time()
                self.perform_object_detection()
                end_time_detection = time.time()
                print("Tiempo de detección de objetos:", end_time_detection - start_time_detection, "segundos")
                

            # Hay píxeles oscuros en la parte inferior de la imagen
            elif np.any(relative_position[:, 1] > 0):
                print("Píxeles oscuros detectados abajo, moviéndose hacia arriba...")
                start_time_move_up = time.time()
                self.client.moveByVelocityAsync(0, 0, -1, 2).join()
                end_time_move_up = time.time()
                print("Tiempo de movimiento hacia arriba:", end_time_move_up - start_time_move_up, "segundos")
                time.sleep(5)
                start_time_detection = time.time()
                self.perform_object_detection()
                end_time_detection = time.time()
                print("Tiempo de detección de objetos:", end_time_detection - start_time_detection, "segundos")
                

            # Hay píxeles oscuros en la parte izquierda de la imagen
            elif np.any(relative_position[:, 0] < 0):
                    print("Píxeles oscuros detectados a la izquierda, moviéndose a la derecha...")
                    start_time_move_right = time.time()
                    self.client.moveByVelocityAsync(1, 0, 0, 1).join()
                    end_time_move_right = time.time()
                    print("Tiempo de movimiento hacia la derecha:", end_time_move_right - start_time_move_right, "segundos")
                    time.sleep(1) 
                    start_time_detection = time.time()
                    self.perform_object_detection()
                    end_time_detection = time.time()
                    print("Tiempo de detección de objetos:", end_time_detection - start_time_detection, "segundos")
                    

            # Hay píxeles oscuros en la parte derecha de la imagen
            elif np.any(relative_position[:, 0] > 0):
                print("Píxeles oscuros detectados a la derecha, moviéndose a la izquierda...")
                start_time_move_left = time.time()
                self.client.moveByVelocityAsync(-1, 0, 0, 1).join()
                end_time_move_left = time.time()
                print("Tiempo de movimiento hacia la izquierda:", end_time_move_left - start_time_move_left, "segundos")
                time.sleep(1)
                start_time_detection = time.time()
                self.perform_object_detection()
                end_time_detection = time.time()
                print("Tiempo de detección de objetos:", end_time_detection - start_time_detection, "segundos")

            # Realizar la detección de objetos nuevamente después del ajuste de posición
            relative_position = self.get_relative_position_of_darkest_pixels()
            distances = np.linalg.norm(relative_position, axis=1)
            close_pixels_indices = np.where(distances < distance_threshold)[0]

        else:
            print("No se detectaron objetos cercanos. No se tomarán medidas evasivas.")
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
            # Realizar la detección de objetos
            start_time_detection = time.time()
            responses = self.client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.Scene, False, False)])
            response = responses[0]
            img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
            img_rgb = img1d.reshape(response.height, response.width, 3)

            # Convertir la imagen a escala de grises
            gray_image = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
            
            
            if len(gray_image.shape) == 2:
                gray_image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2RGB)

            # Valor del pixel mas ocuro
            threshold_value = 0

            # Realizar la detección de objetos si el valor de gris más oscuro es menor que el pixel mas oscuro determinado
            if np.min(gray_image) < threshold_value:
                darkest_pixel_value = np.min(gray_image)
                print("Valor de los píxeles más oscuros:", darkest_pixel_value)
                print("Se detectaron píxeles oscuros. Realizando detección de objetos...")
                self.object_detector.detect_objects(gray_image)
            else:
                print("No se detectaron píxeles oscuros. No se realizará la detección de objetos.")

            end_time_detection = time.time()
            print("Tiempo de detección de objetos:", end_time_detection - start_time_detection, "segundos")

if __name__ == "__main__":
    airsim_wrapper = AirSimWrapper()
    airsim_wrapper.takeoff()
    airsim_wrapper.perform_object_detection()
    airsim_wrapper.fly_to()
    airsim_wrapper.land()
