from nuscenes.nuscenes import NuScenes

nusc = NuScenes(
    version='v1.0-mini',
    dataroot=r'C:\Users\ryanj\OneDrive\바탕 화면\과제\사미용두(개인)\v1.0-mini',
    verbose=True
)

scene = nusc.scene[0]

sample = nusc.get(
    'sample',
    scene['first_sample_token']
)

#---------------------------------------------------------------------
# CAM_FRONT 데이터 가져오기
cam_token = sample['data']['CAM_FRONT']  #정면 카메라의 식별자 get 

cam_data = nusc.get(
    'sample_data',
    cam_token
)

print(cam_data)
print(cam_data['filename'])

from PIL import Image
import matplotlib.pyplot as plt
import os

img_path = os.path.join(
    nusc.dataroot,
    cam_data['filename']
)

print("이미지 경로:", img_path)

img = Image.open(img_path)

plt.imshow(img)    #카메라 데이터 시각화
plt.axis('off')
plt.show()


from nuscenes.utils.data_classes import LidarPointCloud


# LiDAR 데이터 가져오기
lidar_token = sample['data']['LIDAR_TOP']

lidar_data = nusc.get(
    'sample_data',
    lidar_token
)

print(lidar_data['filename'])

# LiDAR 포인트 불러오기
lidar_path = os.path.join(
    nusc.dataroot,
    lidar_data['filename']
)

pc = LidarPointCloud.from_file(lidar_path)  #라이다 파일 읽어오기

print("포인트 개수:", pc.points.shape[1])  #해당 프레임에 찍힌 포인트의 총 개수 출력

from nuscenes.utils.data_classes import RadarPointCloud


# Radar 데이터 가져오기
radar_token = sample['data']['RADAR_FRONT']

radar_data = nusc.get(
    'sample_data',
    radar_token
)

print(radar_data['filename'])

# Radar 포인트 불러오기
radar_path = os.path.join(
    nusc.dataroot,
    radar_data['filename']
)

radar_pc = RadarPointCloud.from_file(radar_path)

print("Radar 포인트 개수:", radar_pc.points.shape[1])  # 레이더 포인트 총 개수 출력


#----------------------------------------------------------------------

# 각 센서의 sample_data 가져오기
cam_data = nusc.get('sample_data', sample['data']['CAM_FRONT'])
lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
radar_data = nusc.get('sample_data', sample['data']['RADAR_FRONT'])

# timestamp 확인
print("Camera :", cam_data['timestamp'])
print("LiDAR  :", lidar_data['timestamp'])
print("Radar  :", radar_data['timestamp'])

# 시간 차이 확인
print("\n시간 차이 (ms)")
print("Camera - LiDAR:",
      (cam_data['timestamp'] - lidar_data['timestamp']) / 1000)

print("Camera - Radar:",
      (cam_data['timestamp'] - radar_data['timestamp']) / 1000)

print("LiDAR - Radar:",
      (lidar_data['timestamp'] - radar_data['timestamp']) / 1000)


#----------------------------------------

import matplotlib.pyplot as plt
import numpy as np

img_path = os.path.join(         #camera
    nusc.dataroot,
    cam_data['filename']
)

img = Image.open(img_path)

plt.figure(figsize=(12, 6))
plt.imshow(img)
plt.title('CAM_FRONT')
plt.axis('off')
plt.show()


points = pc.points   #lidar top view

x = points[0, :]
y = points[1, :]

plt.figure(figsize=(10, 10))
plt.scatter(x, y, s=0.5)
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('LIDAR_TOP - Top View')
plt.axis('equal')
plt.grid()
plt.show()

radar_points = radar_pc.points   # radar top view

x_radar = radar_points[0, :]
y_radar = radar_points[1, :]

plt.figure(figsize=(10, 10))
plt.scatter(x_radar, y_radar, s=20)
plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('RADAR_FRONT - Top View')
plt.axis('equal')
plt.grid()
plt.show()

#-------------------------------------

from pyquaternion import Quaternion
import numpy as np
import matplotlib.pyplot as plt

cam_data = nusc.get(             # 각 센서 데이터 get 
    'sample_data',
    sample['data']['CAM_FRONT']
)

lidar_data = nusc.get(
    'sample_data',
    sample['data']['LIDAR_TOP']
)

radar_data = nusc.get(
    'sample_data',
    sample['data']['RADAR_FRONT']
)


def sensor_to_ego(points, sensor_data):          # 센서 좌표계 -> ego 좌표계 (차량 좌표계)

    calibrated_sensor = nusc.get(
        'calibrated_sensor',
        sensor_data['calibrated_sensor_token']
    )

    rotation = Quaternion(
        calibrated_sensor['rotation']
    ).rotation_matrix

    translation = np.array(
        calibrated_sensor['translation']
    )

    points_ego = rotation @ points + translation.reshape(3, 1)

    return points_ego

lidar_ego = sensor_to_ego(    # lidar -> ego
    pc.points[:3, :],
    lidar_data
)


radar_ego = sensor_to_ego(    # radar -> ego
    radar_pc.points[:3, :],
    radar_data
)

camera_origin = np.zeros((3, 1))   # camera -> ego

camera_ego = sensor_to_ego(
    camera_origin,
    cam_data
)


print("\n===== Sensor → Ego Coordinate =====")   # result
print("\nCamera 센서 위치 (Ego):")
print(camera_ego[:, 0])

print("\nLiDAR 센서 위치 (Ego):")
lidar_calib = nusc.get(
    'calibrated_sensor',
    lidar_data['calibrated_sensor_token']
)
print(lidar_calib['translation'])

print("\nRadar 센서 위치 (Ego):")
radar_calib = nusc.get(
    'calibrated_sensor',
    radar_data['calibrated_sensor_token']
)
print(radar_calib['translation'])


plt.figure(figsize=(10, 10))   # 센서 위치 표시

# LiDAR point cloud
plt.scatter(
    lidar_ego[0, :],
    lidar_ego[1, :],
    s=0.5,
    label='LiDAR'
)

# Radar point cloud
plt.scatter(
    radar_ego[0, :],
    radar_ego[1, :],
    s=30,
    label='Radar'
)

# Camera 센서 위치
plt.scatter(
    camera_ego[0, 0],
    camera_ego[1, 0],
    s=150,
    marker='x',
    label='Camera'
)

# 차량 중심
plt.scatter(
    0,
    0,
    s=100,
    marker='x',
    label='Ego origin'
)

plt.xlabel('X (m)')
plt.ylabel('Y (m)')
plt.title('Camera + LiDAR + Radar in Ego Coordinate')
plt.axis('equal')
plt.grid()
plt.legend()
plt.show()