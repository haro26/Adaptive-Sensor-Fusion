from nuscenes.nuscenes import NuScenes

nusc = NuScenes(
    version='v1.0-mini',
    dataroot=r'C:\Users\ryanj\OneDrive\바탕 화면\과제\사미용두(개인)\v1.0-mini',
    verbose=True
)

print("scene 개수:", len(nusc.scene))


#-----------------------------------------
scene = nusc.scene[0]       # 첫 scene의 첫 sample 가져오기

print("scene 이름:", scene['name'])

sample = nusc.get('sample', scene['first_sample_token'])

for _ in range(29):

    if sample["next"] == "":
        break

    sample = nusc.get(
        "sample",
        sample["next"]
    )

print("sample token:", sample['token'])
print("센서 데이터:")
print(sample['data'].keys())

#------------------------------------------
camera_token = sample['data']['CAM_FRONT']     # 사용할 센서 선택
lidar_token = sample['data']['LIDAR_TOP']
radar_token = sample['data']['RADAR_FRONT']

print("Camera:", camera_token)
print("LiDAR :", lidar_token)
print("Radar :", radar_token)

camera_data = nusc.get('sample_data', camera_token)   # 파일 경로 불러오기
lidar_data = nusc.get('sample_data', lidar_token)
radar_data = nusc.get('sample_data', radar_token)

print("\nCamera 파일:")
print(camera_data['filename'])

print("\nLiDAR 파일:")
print(lidar_data['filename'])

print("\nRadar 파일:")
print(radar_data['filename'])


#------------------------------------------
import numpy as np
from nuscenes.utils.data_classes import LidarPointCloud    # Lidar quality 계산

lidar_path = nusc.get_sample_data_path(lidar_token)
pc = LidarPointCloud.from_file(lidar_path)
points = pc.points
original_num_points = points.shape[1]
print("원본 LiDAR point 수:", original_num_points)

# 0.0 = 정상               # Lidar Degradation (안개로 인한 유효 포인트 감소)
# 0.3 = point 30% 제거
# 0.5 = point 50% 제거
# 0.7 = point 70% 제거

degradation = 0.0
remaining_ratio = 1.0 - degradation   
remaining_num_points = int(original_num_points * remaining_ratio)

if degradation > 0:

    indices = np.random.choice(
        original_num_points,
        remaining_num_points,
        replace=False
    )

    degraded_points = points[:, indices]

else:

    degraded_points = points


current_num_points = degraded_points.shape[1]
lidar_quality = (current_num_points / original_num_points)
lidar_quality = np.clip(lidar_quality,0.0,1.0)

print("현재 LiDAR point 수:", current_num_points)
print("LiDAR degradation:", degradation)
print("LiDAR Quality:", lidar_quality)


#---------------------------------------
import cv2
import numpy as np
from nuscenes.utils.data_classes import LidarPointCloud   # Camera quality 계산

camera_path = nusc.get_sample_data_path(camera_token)
print("Camera path:", camera_path)

import cv2
import numpy as np

image_data = np.fromfile(camera_path,dtype=np.uint8)
image = cv2.imdecode(image_data,cv2.IMREAD_COLOR)

if image is None:
    raise ValueError("Camera 이미지를 불러오지 못했습니다.")

print("Camera image shape:", image.shape)
print("Image shape:", image.shape)


gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # 밝기 계산 (degradation -> 밝기 감소)
mean_brightness = np.mean(gray)
print("평균 밝기:", mean_brightness)

TARGET_BRIGHTNESS = 128.0

brightness_error = abs(mean_brightness - TARGET_BRIGHTNESS)
camera_quality = 1.0 - (brightness_error / TARGET_BRIGHTNESS)
camera_quality = np.clip(camera_quality,0.0,1.0)

print("Camera Quality:", camera_quality)


# 0.0 = 정상
# 0.3 = 밝기 30% 감소
# 0.5 = 밝기 50% 감소
# 0.7 = 밝기 70% 감소

degradation = 0.0
degraded_image = image.astype(np.float32)
degraded_image *= (1.0 - degradation)
degraded_image = np.clip(degraded_image,0,255).astype(np.uint8)
degraded_gray = cv2.cvtColor(degraded_image,cv2.COLOR_BGR2GRAY)
degraded_brightness = np.mean(degraded_gray)
brightness_error = abs(degraded_brightness - TARGET_BRIGHTNESS)
degraded_camera_quality = 1.0 - (brightness_error / TARGET_BRIGHTNESS)
degraded_camera_quality = np.clip(degraded_camera_quality,0.0,1.0)

print("원본 평균 밝기:", mean_brightness)
print("Degraded 평균 밝기:", degraded_brightness)
print("원본 Camera Quality:", camera_quality)
print("Degraded Camera Quality:", degraded_camera_quality)


#-----------------------------------------
import numpy as np
from nuscenes.utils.data_classes import RadarPointCloud    # Radar quality 계산

radar_path = nusc.get_sample_data_path(radar_token)
print("Radar path:", radar_path)

radar_pc = RadarPointCloud.from_file(radar_path)
print("Radar data shape:", radar_pc.points.shape)

num_radar_points = radar_pc.points.shape[1]
print("Radar detection 수:", num_radar_points)


REFERENCE_RADAR_POINTS = num_radar_points
current_radar_points = num_radar_points
radar_quality = (current_radar_points / REFERENCE_RADAR_POINTS)
radar_quality = np.clip(
    radar_quality,
    0.0,
    1.0
)
print("Radar Quality:", radar_quality)

# 0.0 = 정상
# 0.3 = detection 30% 감소
# 0.5 = detection 50% 감소
# 0.7 = detection 70% 감소

degradation = 0.0
remaining_ratio = 1.0 - degradation

remaining_num_points = int(
    num_radar_points * remaining_ratio
)


if degradation > 0:

    indices = np.random.choice(
        num_radar_points,
        remaining_num_points,
        replace=False
    )

    degraded_radar_points = radar_pc.points[:, indices]

else:

    degraded_radar_points = radar_pc.points


current_radar_points = degraded_radar_points.shape[1]

degraded_radar_quality = (
    current_radar_points / num_radar_points
)

degraded_radar_quality = np.clip(
    degraded_radar_quality,
    0.0,
    1.0
)

print("원본 Radar detection 수:", num_radar_points)
print("Degraded Radar detection 수:", current_radar_points)
print("Radar degradation:", degradation)
print("Radar Quality:", degraded_radar_quality)


#-----------------------------------------------
import numpy as np  # quality -> Reliability -> fusion weights


def calculate_reliability_and_weights(
    degraded_camera_quality,
    lidar_quality,
    degraded_radar_quality
):
   

    camera_reliability = np.clip(       # quality -> reliability
        degraded_camera_quality, 0.0, 1.0
    )

    lidar_reliability = np.clip(
        lidar_quality, 0.0, 1.0
    )

    radar_reliability = np.clip(
        degraded_radar_quality, 0.0, 1.0
    )

    total_reliability = (
        camera_reliability
        + lidar_reliability
        + radar_reliability
    )


    if total_reliability == 0:   # reliability -> fusion weight

        camera_weight = 1 / 3
        lidar_weight = 1 / 3
        radar_weight = 1 / 3

    else:

        camera_weight = (
            camera_reliability
            / total_reliability
        )

        lidar_weight = (
            lidar_reliability
            / total_reliability
        )

        radar_weight = (
            radar_reliability
            / total_reliability
        )


    return {
        "camera_reliability": camera_reliability,
        "lidar_reliability": lidar_reliability,
        "radar_reliability": radar_reliability,

        "camera_weight": camera_weight,
        "lidar_weight": lidar_weight,
        "radar_weight": radar_weight
    }

result=calculate_reliability_and_weights(degraded_camera_quality, lidar_quality, degraded_radar_quality)
print(result)


#---------------------------------------
import numpy as np   # current adaptive sensor fusion

def adaptive_fusion(
    camera_detections,
    lidar_detections,
    radar_detections,
    camera_weight,
    lidar_weight,
    radar_weight,
    distance_threshold=2.0
):

    weighted_detections = []  # sensor weight 적용

    for detection in camera_detections:

        d = detection.copy()

        d["weighted_confidence"] = (
            d["confidence"] * camera_weight
        )

        d["sensor"] = "camera"

        weighted_detections.append(d)


    for detection in lidar_detections:

        d = detection.copy()

        d["weighted_confidence"] = (
            d["confidence"] * lidar_weight
        )

        d["sensor"] = "lidar"

        weighted_detections.append(d)


    for detection in radar_detections:

        d = detection.copy()

        d["weighted_confidence"] = (
            d["confidence"] * radar_weight
        )

        d["sensor"] = "radar"

        weighted_detections.append(d)



    fused_detections = [] # 객체별로 detection 묶기

    for detection in weighted_detections:

        matched = False

        for fused in fused_detections:

            # 같은 class인지 확인
            if detection["class"] != fused["class"]:
                continue

            # Ego 좌표계에서 거리 계산
            distance = np.sqrt(
                (detection["x"] - fused["x"]) ** 2
                +
                (detection["y"] - fused["y"]) ** 2
                +
                (detection["z"] - fused["z"]) ** 2
            )

            # 같은 객체라고 판단
            if distance < distance_threshold:

                # 기존 detection의 weighted confidence
                old_score = fused["weighted_confidence"]

                # 현재 detection의 weighted confidence
                new_score = detection["weighted_confidence"]

                total_score = old_score + new_score

                # confidence 기반 위치 가중 평균
                if total_score > 0:

                    fused["x"] = (
                        fused["x"] * old_score
                        +
                        detection["x"] * new_score
                    ) / total_score

                    fused["y"] = (
                        fused["y"] * old_score
                        +
                        detection["y"] * new_score
                    ) / total_score

                    fused["z"] = (
                        fused["z"] * old_score
                        +
                        detection["z"] * new_score
                    ) / total_score

                # confidence 합산
                fused["weighted_confidence"] = (
                    min(1.0, total_score)
                )

                # 어떤 센서들이 해당 객체를 봤는지 기록
                if detection["sensor"] not in fused["sensors"]:

                    fused["sensors"].append(
                        detection["sensor"]
                    )

                matched = True

                break


        # 기존 객체와 매칭되지 않은 경우
        if not matched:

            fused_detections.append({
                "class": detection["class"],
                "x": detection["x"],
                "y": detection["y"],
                "z": detection["z"],
                "weighted_confidence":
                    detection["weighted_confidence"],
                "sensors": [detection["sensor"]]
            })


    return fused_detections



from pyquaternion import Quaternion
import numpy as np


def get_gt_detections_in_ego(nusc, sample):   # nuScenes Ground Truth -> Ego 좌표계

    # sample의 ego pose 가져오기
    lidar_token = sample["data"]["LIDAR_TOP"]

    lidar_data = nusc.get(
        "sample_data",
        lidar_token
    )

    ego_pose = nusc.get(
        "ego_pose",
        lidar_data["ego_pose_token"]
    )

    ego_translation = np.array(
        ego_pose["translation"]
    )

    ego_rotation = Quaternion(
        ego_pose["rotation"]
    )

    detections = []


    for annotation_token in sample["anns"]:   # annotation 확인

        annotation = nusc.get(
            "sample_annotation",
            annotation_token
        )


        global_position = np.array(    # global 위치
            annotation["translation"]
        )

        # Global → Ego
        ego_position = (
            ego_rotation.inverse
            .rotate(
                global_position
                - ego_translation
            )
        )

  
        category = annotation["category_name"]   # class

        if category == "vehicle.car":
            object_class = "car"

        elif category.startswith("vehicle."):
            object_class = "vehicle"

        elif category.startswith("human."):
            object_class = "pedestrian"

        else:
            object_class = category


        detections.append({    # detection 형태로 저장

            "class": object_class,

            "x": float(ego_position[0]),
            "y": float(ego_position[1]),
            "z": float(ego_position[2]),

            # GT이므로 현재는 1.0
            "confidence": 1.0
        })


    return detections


gt_detections = get_gt_detections_in_ego(    # ground truth 가져오기
    nusc,
    sample
)

print("\n========== Ground Truth ==========")

print("객체 수:", len(gt_detections))

for detection in gt_detections[:10]:

    print(
        detection
    )


def create_sensor_detections(    # 센서별 detection 생성
    gt_detections,
    sensor_quality,
    sensor_name,
    position_noise=0.5
):

    detections = []


    for gt in gt_detections:


        detection_probability = sensor_quality    # quality 가 낮을수록 detection 누락

        if np.random.random() > detection_probability:
            continue


        noise_scale = (    # 센서 위치 오차
            position_noise
            * (1.0 - sensor_quality)
        )

        x = (
            gt["x"]
            + np.random.normal(0, noise_scale)
        )

        y = (
            gt["y"]
            + np.random.normal(0, noise_scale)
        )

        z = (
            gt["z"]
            + np.random.normal(0, noise_scale)
        )


        confidence = (        # detection confidence
            sensor_quality
            * np.random.uniform(0.8, 1.0)
        )

        detections.append({

            "class": gt["class"],

            "x": float(x),
            "y": float(y),
            "z": float(z),

            "confidence": float(confidence)
        })


    return detections

camera_detections = create_sensor_detections(gt_detections,degraded_camera_quality,"camera")
lidar_detections = create_sensor_detections(gt_detections,lidar_quality,"lidar")
radar_detections = create_sensor_detections(gt_detections,degraded_radar_quality,"radar")

print("\n========== Sensor Detection ==========")

print("Camera Detection:",len(camera_detections))
print("LiDAR Detection:",len(lidar_detections))
print("Radar Detection:",len(radar_detections))

fused_result = adaptive_fusion(   # adaptive fusion

    camera_detections,
    lidar_detections,
    radar_detections,

    result["camera_weight"],
    result["lidar_weight"],
    result["radar_weight"]
)
print("\n========== Adaptive Fusion Result ==========")
print("Fused Object 수:",len(fused_result))

for detection in fused_result:

    print(
        f"Class: {detection['class']}, "
        f"Position: "
        f"({detection['x']:.2f}, "
        f"{detection['y']:.2f}, "
        f"{detection['z']:.2f}), "
        f"Confidence: "
        f"{detection['weighted_confidence']:.3f}, "
        f"Sensors: "
        f"{detection['sensors']}"
    )

camera_only = 0
lidar_radar_only = 0
all_three = 0

for detection in fused_result:

    sensors = set(detection["sensors"])

    if "camera" in sensors:
        camera_only +=1
    if sensors == {"lidar", "radar"}:
        lidar_radar_only +=1
    if sensors == {"camera", "lidar", "radar"}:
        all_three +=1 

print("camera 포함:", camera_only)
print("radar + lidar만:", lidar_radar_only)
print("3개 센서 모두:", all_three)


# ==============================
# Predictive Adaptive Fusion
# ==============================

ENVIRONMENT_RANGES = {"Tunnel": range(10, 16),"Raining": range(20, 26),"Fog": range(30, 36)}
FUTURE_HORIZON = 2.0

def get_sample_index(nusc, sample):

    scene = nusc.get("scene",sample["scene_token"])
    sample_token = scene["first_sample_token"]
    sample_index = 0


    while sample_token != sample["token"]:

        current_sample = nusc.get("sample",sample_token)
        sample_token = current_sample["next"]
        sample_index += 1


        if sample_token == "":
            break
    return sample_index


def predict_future_environment(nusc, current_sample):

    future_samples = []

    current_timestamp = current_sample["timestamp"]

    current_index = get_sample_index(
        nusc,
        current_sample
    )

    future_sample = current_sample
    future_index = current_index

    while future_sample["next"] != "":

        next_token = future_sample["next"]

        future_sample = nusc.get(
            "sample",
            next_token
        )

        future_index += 1

        elapsed_time = (
            future_sample["timestamp"]
            - current_timestamp
        ) / 1_000_000.0

        if elapsed_time > FUTURE_HORIZON:
            break

        future_samples.append(
            (future_sample, future_index)
        )


    future_environment = "Normal"

    for future_sample, sample_index in future_samples:

        if sample_index in ENVIRONMENT_RANGES["Tunnel"]:
            future_environment = "Tunnel"
            break

        elif sample_index in ENVIRONMENT_RANGES["Raining"]:
            future_environment = "Raining"
            break

        elif sample_index in ENVIRONMENT_RANGES["Fog"]:
            future_environment = "Fog"
            break


    return future_environment

print("현재 sample index:", get_sample_index(nusc, sample))
future_environment = predict_future_environment(nusc, sample)
print("\n=========Future Environment===========")
print("Predict Future Environment: ", future_environment)

if future_environment == "Tunnel":           # camera
        predicted_camera_quality = 0.23 
        predicted_lidar_quality = 1.0
        predicted_radar_quality = 1.0
elif future_environment == "Raining":        # radar
        predicted_camera_quality = 0.86
        predicted_lidar_quality = 1.0
        predicted_radar_quality = 0.29
elif future_environment == "Fog":            # lidar
        predicted_camera_quality = 0.86
        predicted_lidar_quality = 0.29
        predicted_radar_quality = 1.0
else:
    predicted_camera_quality=1.0
    predicted_lidar_quality=1.0
    predicted_radar_quality=1.0
    

predicted_result = calculate_reliability_and_weights(predicted_camera_quality,predicted_lidar_quality,predicted_radar_quality)

predictive_camera_detections = create_sensor_detections(gt_detections,predicted_camera_quality,"camera")
predictive_lidar_detections = create_sensor_detections(gt_detections,predicted_lidar_quality,"lidar")
predictive_radar_detections = create_sensor_detections(gt_detections,predicted_radar_quality,"radar")

predictive_fused_result = adaptive_fusion(
    predictive_camera_detections,
    predictive_lidar_detections,
    predictive_radar_detections,
    predicted_result["camera_weight"],
    predicted_result["lidar_weight"],
    predicted_result["radar_weight"]
)

print("\n=======Predictive Result=======")

print("Future Environment: ",future_environment)
print("Predicted Camera Quality: ",predicted_camera_quality)
print("Predicted Lidar Quality: ",predicted_lidar_quality)
print("Predicted Radar Quality: ",predicted_radar_quality)

print("Predicted Camera Weight: ",predicted_result["camera_weight"])
print("Predicted Lidar Weight: ",predicted_result["lidar_weight"])
print("Predicted Radar Weight: ",predicted_result["radar_weight"])

print("\n========== Predictive Adaptive Fusion Result ==========")
print("Predictive Fused Object 수:",len(predictive_fused_result))

for detection in predictive_fused_result:

    print(
        f"Class: {detection['class']}, "
        f"Position: "
        f"({detection['x']:.2f}, "
        f"{detection['y']:.2f}, "
        f"{detection['z']:.2f}), "
        f"Confidence: "
        f"{detection['weighted_confidence']:.3f}, "
        f"Sensors: "
        f"{detection['sensors']}"
    )

camera_only = 0
lidar_radar_only = 0
all_three = 0

for detection in predictive_fused_result:

    sensors = set(detection["sensors"])

    if "camera" in sensors:
        camera_only +=1
    if sensors == {"lidar", "radar"}:
        lidar_radar_only +=1
    if sensors == {"camera", "lidar", "radar"}:
        all_three +=1 

print("camera 포함:", camera_only)
print("radar + lidar만:", lidar_radar_only)
print("3개 센서 모두:", all_three)