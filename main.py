import json
import os.path
import os
import subprocess
import numpy as np


num_data = 10
dataPath = "/home/chen/data/Example_Data"
OPENMVG_SFM_BIN = "/home/chen/openMVG_Build/Linux-x86_64-RELEASE"
# Indicate the openMVG camera sensor width directory
CAMERA_SENSOR_WIDTH_DIRECTORY = "/home/chen/openMVG/src/software/SfM" + "/../../openMVG/exif/sensor_width_database"
fileNames = []
gt = []
obs = []


def read_data():
    for i in range(num_data):
        if (i + 1) < 10:
            fileNames.append("frame0000" + str(i + 1) + ".pose.txt")
        elif (i + 1) < 100:
            fileNames.append("frame000" + str(i + 1) + ".pose.txt")
        for name in fileNames:
            filePath = os.path.join(dataPath, name)
        with open(filePath, 'r') as pose:
            isGT = 1
            for line in pose.readlines():
                if isGT:
                    temp = line.split()
                    for j in range(len(temp)):
                        temp[j] = float(temp[j])
                    gt.append(temp)
                    isGT = 0
                else:
                    temp = line.split()
                    for j in range(len(temp)):
                        temp[j] = float(temp[j])
                    obs.append(temp)


def read_data_new():
    with open("./new_data/dataset_test.txt", 'r') as file:
        for line in file.readlines():
            temp = line.split()
            if "seq3" in temp[0]:
                gt.append(temp)

    for i in gt:
        i[0] = int(i[0][10:15])
        for j in range(1, 8):
            i[j] = float(i[j])

    for a in range(len(gt) - 1):
        max_index = a
        for b in range(max_index + 1, len(gt)):
            if gt[b] < gt[max_index]:
                max_index = b
        gt[a], gt[max_index] = gt[max_index], gt[a]

    for data in gt:
        data.pop(0)
        # print(data)


def quaternion_to_rotation_matrix(quat, position):
    q = quat.copy()
    n = np.dot(q, q)
    if n < np.finfo(q.dtype).eps:
        return np.identity(4)
    q = q * np.sqrt(2.0 / n)
    q = np.outer(q, q)
    rot_matrix = np.array(
        [[1.0 - q[2, 2] - q[3, 3], q[1, 2] + q[3, 0], q[1, 3] - q[2, 0], position[0]],
         [q[1, 2] - q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] + q[1, 0], position[1]],
         [q[1, 3] + q[2, 0], q[2, 3] - q[1, 0], 1.0 - q[1, 1] - q[2, 2], position[2]]],
        dtype=q.dtype)
    return rot_matrix


read_data_new()
gt_file = open("./GT_data/00.txt", 'w+')
for data in gt:
    translation = np.asarray([data[0], data[1], data[2]])
    rotation_quaternion = np.asarray([data[3], data[4], data[5], data[6]])
    mat = quaternion_to_rotation_matrix(rotation_quaternion, translation)
    print(str(mat[0][0]) + " " + str(mat[0][1]) + " " + str(mat[0][2]) + " " + str(mat[0][3]) + " " + str(mat[1][0]) +
          " " + str(mat[1][1]) + " " + str(mat[1][2]) + " " + str(mat[1][3]) + " " + str(mat[2][0]) + " " +
          str(mat[2][1]) + " " + str(mat[2][2]) + " " + str(mat[2][3]), file=gt_file)
gt_file.close()

intrinsics_1 = {
    "key": 0,
    "value": {
        "polymorphic_id": 2147483649,
        "polymorphic_name": "pinhole_radial_k3",
        "ptr_wrapper": {
            "id": 2147483660,
            "data": {
                "width": 1920,
                "height": 1080,
                "focal_length": 3141.2625,
                "principal_point": [
                    960.0,
                    540.0
                ],
                "disto_k3": [
                    0.0,
                    0.0,
                    0.0
                ]
            }
        }
    }
}

intrinsics_array = [intrinsics_1]


def get_parent_dir(directory):
    return os.path.dirname(directory)


os.chdir(os.path.dirname(os.path.abspath(__file__)))
ground_truth_dir = os.path.abspath("./GT_data")
input_eval_dir = os.path.abspath("./images")
# Checkout an OpenMVG image dataset with Git
if not os.path.exists(input_eval_dir):
    print("no image path")

output_eval_dir = os.path.join(get_parent_dir(input_eval_dir), "test_out")
if not os.path.exists(output_eval_dir):
    os.mkdir(output_eval_dir)

input_dir = input_eval_dir
output_dir = output_eval_dir
print("Using input dir  : ", input_dir)
print("      output_dir : ", output_dir)

matches_dir = os.path.join(output_dir, "matches")
camera_file_params = os.path.join(CAMERA_SENSOR_WIDTH_DIRECTORY, "sensor_width_camera_database.txt")

if not os.path.exists(matches_dir):
    os.mkdir(matches_dir)

print("1. Intrinsics analysis")
pIntrisics = subprocess.Popen(
    [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_SfMInit_ImageListingFromKnownPoses"), "-i", input_dir, "-g", ground_truth_dir, "-t", "5", "-o", matches_dir])
pIntrisics.wait()

with open("./test_out/matches/sfm_data.json", 'r') as f:
    data = json.load(f)
    data["intrinsics"] = intrinsics_array
    for frames in data["views"]:
        frames["value"]["ptr_wrapper"]["data"]["id_intrinsic"] = 0

os.remove("./test_out/matches/sfm_data.json")
with open("./test_out/matches/sfm_data.json", "w") as f:
    json.dump(data, f, indent=4)

print("2. Compute features")
pFeatures = subprocess.Popen(
    [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeFeatures"), "-i", matches_dir + "/sfm_data.json", "-o",
     matches_dir, "-m", "SIFT", "-f", "1"])
pFeatures.wait()

print("3. Compute matches")
pMatches = subprocess.Popen(
    [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeMatches"), "-i", matches_dir + "/sfm_data.json", "-o",
     matches_dir + "/matches.putative.bin", "-f", "1", "-n", "ANNL2"])
pMatches.wait()

print("4. Filter matches")
pFiltering = subprocess.Popen(
    [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_GeometricFilter"), "-i", matches_dir + "/sfm_data.json", "-m",
     matches_dir + "/matches.putative.bin", "-g", "f", "-o", matches_dir + "/matches.f.bin"])
pFiltering.wait()

print("5. Output Camera Poses")
pCameraPose = subprocess.Popen(
    [os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ExportCameraFrustums"), "-i", matches_dir + "/sfm_data.json", "-o",
     matches_dir + "camera_pose.ply"])
pCameraPose.wait()

print("5. Structure from Known Poses (robust triangulation)")
pRecons = subprocess.Popen([os.path.join(OPENMVG_SFM_BIN, "openMVG_main_ComputeStructureFromKnownPoses"), "-i",
                            matches_dir + "/sfm_data.json", "-m", matches_dir, "-b", "-o",
                            os.path.join(output_eval_dir, "robust.ply")])
pRecons.wait()
