import os
import sys

__file__ = r'C:\Users\Nick\Desktop\capstone\TrainYourOwnYOLO\3_Inference\DetectorNG.py'


def get_parent_dir(n=1):
    """ returns the n-th parent dicrectory of the current
    working directory """
    current_path = os.path.dirname(os.path.abspath(__file__))
    for k in range(n):
        current_path = os.path.dirname(current_path)
    return current_path


src_path = os.path.join(get_parent_dir(1), "2_Training", "src")
utils_path = os.path.join(get_parent_dir(1), "Utils")

sys.path.append(src_path)
sys.path.append(utils_path)
sys.path.append(get_parent_dir(1))

from keras_yolo3.yolo import YOLO
from timeit import default_timer as timer
from utils import detect_object
import pandas as pd
import numpy as np
from capture_image_webcam import take_picture

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Set up folder names for default values
data_folder = os.path.join(get_parent_dir(n=1), "Data")

image_folder = os.path.join(data_folder, "Source_Images")

image_test_folder = os.path.join(image_folder, "Test_Images")

detection_results_folder = os.path.join(image_folder, "Test_Image_Detection_Results")
detection_results_file = os.path.join(detection_results_folder, "Detection_Results.csv")

model_folder = os.path.join(data_folder, "Model_Weights")

model_weights = os.path.join(model_folder, "trained_weights_final.h5")
model_classes = os.path.join(model_folder, "data_classes.txt")

anchors_path = os.path.join(src_path, "keras_yolo3", "model_data", "yolo_anchors.txt")

confidence_score = 0.25

save_img = True

file_types = '.jpg'

# Split images and videos
output_path = detection_results_folder

if not os.path.exists(output_path):
    os.makedirs(output_path)

# define YOLO detector
yolo = YOLO(
    **{
        "model_path": model_weights,
        "anchors_path": anchors_path,
        "classes_path": model_classes,
        "score": confidence_score,
        "gpu_num": 1,
        "model_image_size": (416, 416),
    }
)

# Make a dataframe for the prediction outputs
out_df = pd.DataFrame(
    columns=[
        "image",
        "image_path",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "label",
        "confidence",
        "x_size",
        "y_size",
    ]
)



KNOWN_WIDTH = 3.5

focalLength = 1430


def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
	return (knownWidth * focalLength) / perWidth


# labels to draw on images
class_file = open(model_classes, "r")
input_labels = [line.rstrip("\n") for line in class_file.readlines()]
print("Found {} input labels: {} ...".format(len(input_labels), input_labels))

result = 'c'
while result == 'c':
    result = input("Press 'c' to Get a Cone Prediction...: ")
    
    # start the time
    start = timer()
    
    # This is for taking pics
    take_picture(image_test_folder)
    img_path = os.path.join(image_test_folder, 'robot_fov.jpg')
    
    # MAKE PREDICTION
    print(img_path)
    prediction, image = detect_object(
        yolo,
        img_path,
        save_img=save_img,
        save_img_path=detection_results_folder,
        postfix='_cone',
    )
    
    distance_away = distance_to_camera(KNOWN_WIDTH, focalLength, (prediction[0][2]-prediction[0][0]))
    
    
    # save predictions to csv
    y_size, x_size, _ = np.array(image).shape
    for single_prediction in prediction:
        out_df = out_df.append(
            pd.DataFrame(
                [
                    [
                        os.path.basename(img_path.rstrip("\n")),
                        img_path.rstrip("\n"),
                    ]
                    + single_prediction
                    + [x_size, y_size]
                ],
                columns=[
                    "image",
                    "image_path",
                    "xmin",
                    "ymin",
                    "xmax",
                    "ymax",
                    "label",
                    "confidence",
                    "x_size",
                    "y_size",
                ],
            )
        )
    end = timer()
    
    print(f'\nSUCCESSFULLY PROCESSED 1 IMAGE IN {round(end-start, 2)} SECONDS')
    print(f'\nTHE CONE IS {round(distance_away, 2)} INCHES AWAY FROM ROBOT')
    print('------------------------------------------------------------------')
    
    
    
out_df.to_csv(detection_results_file, index=False)
    
# Close the current yolo session
yolo.close_session()
