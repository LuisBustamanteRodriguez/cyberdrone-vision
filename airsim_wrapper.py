import airsim
import numpy as np
import cv2


# class ObjectDetector:
#     def __init__(self):
#         self.model = YOLO("yolov8l.pt")
#         self.model.conf = 0.5

    # def detect_objects(self, image):
    #     results = self.model(image)
    #     return results

class AirSimWrapper:
    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
    

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

    def get_position(self, object_name):
        query_string = object_name + ".*"
        object_names_ue = []
        while len(object_names_ue) == 0:
            object_names_ue = self.client.simListSceneObjects(query_string)
        pose = self.client.simGetObjectPose(object_names_ue[0])
        return [pose.position.x_val, pose.position.y_val, pose.position.z_val]

    def perform_object_detection(self):
        responses = self.client.simGetImages([airsim.ImageRequest(0, airsim.ImageType.Scene, False, False)])
        response = responses[0]
        img1d = np.fromstring(response.image_data_uint8, dtype=np.uint8)
        img_rgb = img1d.reshape(response.height, response.width, 3)
        image = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)
        results = self.object_detector.detect_objects(image)

        if results:
            class_names = self.object_detector.model.names  # Obtener los nombres de las clases del modelo
            if not class_names:
                print("Error: No se pudieron cargar los nombres de las clases.")
                return

            for detection in results[0]:
                box = detection[:4] if len(detection) >= 4 else [0, 0, 0, 0]
                conf = detection[4] if len(detection) >= 5 else 0
                cls = int(detection[5]) if len(detection) >= 6 else -1
            
                if 0 <= cls < len(class_names):
                    label = f"{class_names[int(cls)]}: {conf:.2f}"
                else:
                    print(f"Índice de Clase no válido: {cls}")  # Depuración
                    label = f"Objeto Desconocido: {conf:.2f}"

            x1, y1, x2, y2 = [int(coord) for coord in box]

            object_center_x = (x1 + x2) / 2
            image_center_x = image.shape[1] / 2
            distance_factor = 100  # Ajusta este valor según la escala y la distancia focal de tu cámara

            # Calcula la posición relativa del objeto
            if object_center_x < image_center_x - 50:
                position = "izquierda"
            elif object_center_x > image_center_x + 50:
                position = "derecha"
            else:
                position = "centro"

            # Calcula la distancia estimada basada en el tamaño del objeto en la imagen
            object_width = x2 - x1
            object_height = y2 - y1
            object_size = max(object_width, object_height)
            estimated_distance = distance_factor / object_size if object_size > 0 else float('inf')

            print(f"{label} - Posición: {position}, Distancia estimada: {estimated_distance:.2f} metros")
        else:
            print("No se detectaron objetos en la imagen.")

if __name__ == "__main__":
    airsim_wrapper = AirSimWrapper()
    airsim_wrapper.takeoff()
    airsim_wrapper.perform_object_detection()
    airsim_wrapper.land()