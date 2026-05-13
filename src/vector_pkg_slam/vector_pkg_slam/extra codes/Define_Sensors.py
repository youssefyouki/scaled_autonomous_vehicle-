# Setup_Devices.py
import math

#####################################################################
# Define Car Sensor Slots
Center_Slot = [0,0,0]
Fornt_Slot = [3.79, 0, 0.142]
Top_Slot = [1.11, 0, 1.16]
Rear_Slot = [-1.05, 0, 0.2]
#####################################################################

#####################################################################
# Define sensor categories
Localization_Sensors = ['encoder', 'imu']
Perception_Sensors = ['zed_camera', 'pi_camera', 'rplidar']
Displays = []
#####################################################################

#####################################################################
# ZED2-like parameters
ZED2_CONFIG = {
    'width': 1280,
    'height': 720,
    'fov': 1.9199,       # radians (~110°)
    'focal_length_mm': 2.12,
    'baseline': 0.12,  # 12 cm between cameras
    'aperture': 1.8,
    "exposure": 0.6,
    "noise": 0.002
}
#####################################################################

#####################################################################
def setup_sensors(car, timestep):
    devices = {}

    print("\n🔧 Initializing sensors...")

    for sensor_name in Localization_Sensors + Perception_Sensors + Displays:
        print(sensor_name)

        # --- Localization Sensors ---
        if sensor_name == 'encoder':
            print("✅ encoder enabled.")

        elif sensor_name == 'imu':
            print("✅ IMU enabled.")

        # --- Perception Sensors ---
        elif sensor_name == 'zed_camera':
            print("✅ zed_camera enabled.")
        
        elif sensor_name == 'pi_camera':
            print("✅ pi_camera enabled.")
            
        elif sensor_name == 'rplidar':
            print("✅ rplidar enabled.")
       
        elif sensor_name == 'gps':
            print("✅ gps enabled.")
 

    print("✅ Sensor setup complete.\n")
    return devices
#####################################################################
