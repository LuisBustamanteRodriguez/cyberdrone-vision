import airsim
import numpy as np
import cv2
from ultralytics import YOLO


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

    def takeoff(self):
        self.client.takeoffAsync().join()

    def land(self):
        self.client.landAsync().join()

    def get_drone_position(self):
        pose = self.client.simGetVehiclePose()
        return [pose.position.x_val, pose.position.y_val, pose.position.z_val]

    def fly_to(self, point):
        if point[2] > 0:
            self.client.moveToPositionAsync(point[0], point[1], -point[2], 5).join()
        else:
            self.client.moveToPositionAsync(point[0], point[1], point[2], 5).join()

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

    def get_drone_position(self, darkest_pixels_position):
        pose = self.client.simGetVehiclePose()
        drone_position = np.array([pose.position.x_val, pose.position.y_val, pose.position.z_val])
        drone_position = np.tile(drone_position, (darkest_pixels_position.shape[0], 1))  # Expandir para que tenga la misma forma que darkest_pixels_position
        return drone_position[:, :2]  # Ajustar la forma de drone_position para que coincida con darkest_pixels_position


    def process_image(self, image):
        # Convertir la imagen a escala de grises
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Encontrar los píxeles más oscuros
        darkest_pixel_value = np.min(gray_image)
        darkest_pixels = np.where(gray_image == darkest_pixel_value)
        
        # Convertir coordenadas de píxeles a formato adecuado
        darkest_pixels_position = np.column_stack((darkest_pixels[0], darkest_pixels[1]))
        print("Darkest pixels position shape:", darkest_pixels_position.shape)
        
        # Obtener la posición relativa de los píxeles más oscuros respecto al dron
        drone_position = self.get_drone_position(darkest_pixels_position)  # Pasar darkest_pixels_position como argumento
        relative_position = darkest_pixels_position - drone_position

        # Tomar acciones para evitar colisiones basadas en la posición relativa de los píxeles más oscuros
        self.avoid_collision(relative_position, drone_position)  # Pasar drone_position como argumento

        # Realizar detección de objetos utilizando YOLOv8l
        results = self.object_detector.detect_objects(image)


    def avoid_collision(self, relative_position, drone_position):
        print("Relative position shape:", relative_position.shape)

        # Supongamos que si hay píxeles oscuros en la parte superior de la imagen, el dron debería subir
        if np.any(relative_position[:, 1] < 0):
            print("Píxeles oscuros detectados arriba, subiendo...")
            self.client.moveByVelocityAsync(0, 0, -1, 1).join()  # Hacer que el dron suba

        # Supongamos que si hay píxeles oscuros en la parte inferior de la imagen, el dron debería bajar
        elif np.any(relative_position[:, 1] > 0):
            print("Píxeles oscuros detectados abajo, bajando...")
            self.client.moveByVelocityAsync(0, 0, 1, 1).join()  # Hacer que el dron baje

        # Supongamos que si hay píxeles oscuros en la parte izquierda de la imagen, el dron debería moverse a la izquierda
        if np.any(relative_position[:, 0] < 0):
            print("Píxeles oscuros detectados a la izquierda, moviéndose a la izquierda...")
            self.client.moveByVelocityAsync(-1, 0, 0, 1).join()  # Hacer que el dron se mueva a la izquierda

        # Supongamos que si hay píxeles oscuros en la parte derecha de la imagen, el dron debería moverse a la derecha
        elif np.any(relative_position[:, 0] > 0):
            print("Píxeles oscuros detectados a la derecha, moviéndose a la derecha...")
            self.client.moveByVelocityAsync(1, 0, 0, 1).join()  # Hacer que el dron se mueva a la derecha

    def perform_object_detection(self):
        responses = self.client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.Scene, False, False)])
        response = responses[0]
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(response.height, response.width, 3)

        image = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        self.process_image(image)

if __name__ == "__main__":
    airsim_wrapper = AirSimWrapper()
    airsim_wrapper.takeoff()
    airsim_wrapper.perform_object_detection()
    airsim_wrapper.land()
