# import streamlit as st
# import cv2
# import numpy as np
# import tensorflow as tf
#
# st.title("Object Detection with Streamlit")
#
# # Load MobileNet SSD model and configuration
# model_path = 'pages/models/ssd_mobilenet_v2_320_320/ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model/saved_model.pb'
# # config_path = 'pages/models/ssd_mobilenet_v2_320_320/ssd_mobilenet_v2_320x320_coco17_tpu-8/pipeline.config'
# #
# # net = cv2.dnn.readNetFromTensorflow(model_path + '/saved_model.pb', config_path)
#
# model = tf.saved_model.load(model_path)
#
# cap = cv2.VideoCapture(0)
#
# if not cap.isOpened():
#     st.error("Error: Could not open webcam.")
# else:
#     webcam_success_message = st.success("Webcam is active.")
#
# webcam_placeholder = st.empty()
#
# while cap.isOpened():
#     ret, frame = cap.read()
#
#     if not ret:
#         st.warning("Warning: Could not read a frame from the webcam.")
#         break
#
#     webcam_placeholder.image(frame, channels="BGR", use_column_width=True)
#
# cap.release()

from torchvision.models import detection
import pickle
import torch
import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'street sign', 'stop sign', 'parking meter', 'bench',
    'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
    'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses', 'handbag', 'tie', 'suitcase',
    'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'plate', 'wine glass', 'cup',
    'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
    'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'mirror', 'dining table', 'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'blender', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
    'hair brush'
]

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

MODELS = {
    "frcnn-resnet": detection.fasterrcnn_resnet50_fpn,
    "frcnn-mobilenet": detection.fasterrcnn_mobilenet_v3_large_320_fpn,
    "retinanet": detection.retinanet_resnet50_fpn
}

model = MODELS["frcnn-resnet"](pretrained=True, progress=True,
                               num_classes=len(CLASSES), pretrained_backbone=True).to(DEVICE)
model.eval()

st.title("Object Detection with Streamlit")

image = st.file_uploader("Please upload a photo of your face", accept_multiple_files=False, type=["jpg", "jpeg", "png"])

image_placeholder = st.empty
if image:
    # image_placeholder = st.image(image, channels="BGR", use_column_width=True)

    image = Image.open(image)
    image = ImageOps.exif_transpose(image)

    image = np.array(image)

    orig = image.copy()
    # orig = ImageOps.exif_transpose(orig)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.transpose((2, 0, 1))

    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    image = torch.FloatTensor(image)

    image = image.to(DEVICE)
    detections = model(image)[0]

    # loop over the detections
    for i in range(0, len(detections["boxes"])):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections["scores"][i]
        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # extract the index of the class label from the detections,
            # then compute the (x, y)-coordinates of the bounding box
            # for the object
            idx = int(detections["labels"][i])
            box = detections["boxes"][i].detach().cpu().numpy()
            (startX, startY, endX, endY) = box.astype("int")
            # display the prediction to our terminal
            label = "{}: {:.2f}%".format(CLASSES[idx-1], confidence * 100)
            print("[INFO] {}".format(label))
            st.write(label)
            # draw the bounding box and label on the image
            cv2.rectangle(orig, (startX, startY), (endX, endY),
                          COLORS[idx], 2)
            y = startY - 15 if startY - 15 > 15 else startY + 15
            cv2.putText(orig, label, (startX, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS[idx], 4)


    image_placeholder = st.image(orig, channels="RGB", use_column_width=True)