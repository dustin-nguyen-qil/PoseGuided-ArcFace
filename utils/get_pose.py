"""
    Get pitch and yaw angles from faces of an image
    Compute Pose and perform quantization
"""

# To download dlib, we first must pip install cmake, then pip install dlib
# pose angles are between -90 to +90
# In this repo, the path of photos_all_faces is read and get_xyz() function calculates and return the yaw, pitch and landmark points
# The repo uses the landmarks model stored in Landmarks folder to get teh face keypoints
# The landmark points are returned as x, y pairs of:
# 1. Nose tip
# 2. Chin
# 3. Left eye left corner
# 4. Right eye right corner
# 5. Left Mouth corner
# 6. Right Mouth corner
# These face yaw, pitch and landmark pints are saved in face_angles.json file in Face_information directory

import dlib
import cv2
import math
import numpy as np
import os
import json


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('Landmarks/shape_predictor_68_face_landmarks.dat')


def get_xyz(path):
    # Read Image
    # image = cv2.imread("Test_images/b_cam_0_porR_00.jpg")
    image = cv2.imread(path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    data = np.asarray(gray)
    rects = detector(data, 1)

    #making an assumption on the angles here
    if len(rects) == 0:
        # return np.nan, np.nan, np.nan
        return 0, 0, np.array([(0, 0),  # Nose tip
        (0, 0),  # Chin
        (0, 0),  # Left eye left corner
        (0, 0),  # Right eye right corne
        (0, 0),  # Left Mouth corner
        (0, 0)  # Right mouth corner
    ], dtype="double")

    shape = predictor(data, rects[0])

    # dim = (256, 256)
    #image = cv2.imread("Test_images/b_cam_0_porF_00.jpg")
    image = cv2.imread(path)
    # im = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    size = gray.shape

    # 2D image points. If you change the image, you need to change vector
    image_points = np.array([
        (shape.part(30).x, shape.part(30).y),  # Nose tip
        (shape.part(8).x, shape.part(8).y),  # Chin
        (shape.part(36).x, shape.part(36).y),  # Left eye left corner
        (shape.part(45).x, shape.part(45).y),  # Right eye right corne
        (shape.part(48).x, shape.part(48).y),  # Left Mouth corner
        (shape.part(54).x, shape.part(54).y)  # Right mouth corner
    ], dtype="double")

    # 3D model points.
    model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),  # Right eye right corne
        (-150.0, -150.0, -125.0),  # Left Mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner
    ])

    # Camera internals
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    # print("Camera Matrix :\n {0}".format(camera_matrix))
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs,
                                                                flags=cv2.SOLVEPNP_ITERATIVE)
    mdists = np.zeros((4, 1), dtype=np.float64)

    # calculating angle
    rmat, jac = cv2.Rodrigues(rotation_vector)
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)

    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector,
                                                     translation_vector, camera_matrix, dist_coeffs)
    for p in image_points:
        cv2.circle(image, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)

    rvec_matrix = cv2.Rodrigues(rotation_vector)[0]

    proj_matrix = np.hstack((rvec_matrix, translation_vector))

    eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]

    pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]

    pitch = math.degrees(math.asin(math.sin(pitch)))
    roll = -math.degrees(math.asin(math.sin(roll)))
    yaw = math.degrees(math.asin(math.sin(yaw)))

    return pitch, yaw, image_points


path = 'photos_all_faces/'
image_files = [f for f in os.listdir(path) if f.endswith('.jpg')]
face_angles_list=[]

for i in image_files:
    p = os.path.join(path, i)
    pitch, yaw, image_points = get_xyz(p)

    # {"img_path": path, "id": id, "yaw": yaw, "pitch": pitch}
    dictionary = dict()
    dictionary['img_path'] = p
    dictionary['id'] = i.replace('.jpg','')
    dictionary['yaw'] = yaw
    dictionary['pitch'] = pitch
    dictionary['landmark_points'] = image_points.tolist()

    face_angles_list.append(dictionary)

    # print(p, get_xyz(p))

print(face_angles_list)

# store the info contained in teh dict as json
json_object = json.dumps(face_angles_list, indent=2)
with open("Face_information/face_angles.json", "w") as outfile:
    json.dump(json_object, outfile)

