import airsim
import numpy as np

class AirSimWrapper:
    def __init__(self):
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.depth_image = None  
        self.object_threshold = 1000  # 1 metro
        self.drone_position = [0, 0, 0]  

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

    def get_depth_image(self):
        response = self.client.simGetImages([
            airsim.ImageRequest("1", airsim.ImageType.DepthPerspective, True, False) 
        ])
        depth_response = response[0]

        depth_img_in_meters = airsim.list_to_2d_float_array(depth_response.image_data_float, depth_response.width, depth_response.height)
        depth_img_in_meters = depth_img_in_meters.reshape(depth_response.height, depth_response.width, 1)
        depth_img_in_millimeters = depth_img_in_meters * 1000

        self.depth_image = depth_img_in_millimeters
        
        return depth_img_in_millimeters

    def process_depth_image(self):
        depth_image = self.get_depth_image()  
        closest_object_distance = np.inf  

        for y in range(depth_image.shape[0]):  
            for x in range(depth_image.shape[1]):  
                depth_value = depth_image[y, x]  

                if depth_value > 0:  

                    distance = depth_value / 1000  

                    if distance < closest_object_distance:
                        closest_object_distance = distance

        return closest_object_distance


if __name__ == "__main__":
    airsim_wrapper = AirSimWrapper()
    airsim_wrapper.takeoff()
    airsim_wrapper.fly_to()
    airsim_wrapper.land()
