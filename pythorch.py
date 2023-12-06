from torchvision.models import detection
import pickle
import torch
import cv2
import numpy as np

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

image = cv2.imread('C:/Users/pc-l/Pictures/Kingának/image_67215105.jpg')
cv2.imshow('Image', image)
cv2.waitKey(0)
orig = image.copy()

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image.transpose((2, 0, 1))

image = np.expand_dims(image, axis=0)
image = image / 255.0
image = torch.FloatTensor(image)

image = image.to(DEVICE)
detections = model(image)[0]

# import torch
# import torchvision.transforms as T
# from torchvision.models.detection import fasterrcnn_resnet50_fpn
# from PIL import Image, ImageDraw
# import cv2
# import numpy as np
#
# # Load pre-trained Faster R-CNN model
# model = fasterrcnn_resnet50_fpn(pretrained=True)
# model.eval()
#
# # Define the transformation to be applied to the input image
# transform = T.Compose([T.ToTensor()])
#
# # COCO dataset categories
# COCO_INSTANCE_CATEGORY_NAMES = [
#     'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
#     'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
#     'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
#     'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
#     'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
#     'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
#     'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
#     'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
#     'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
#     'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
#     'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
#     'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
#     'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
#     'hair drier', 'toothbrush'
# ]
#
# # Open webcam
# cap = cv2.VideoCapture(0)
#
# while True:
#     # Capture frame from the webcam
#     ret, frame = cap.read()
#
#     # Convert frame to PIL Image
#     image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#
#     # Apply transformation to the image
#     input_tensor = transform(image).unsqueeze(0)
#
#     # Perform inference
#     with torch.no_grad():
#         predictions = model(input_tensor)
#
#     # Display predictions on the frame
#     draw = ImageDraw.Draw(image)
#
#     for score, label, box in zip(predictions[0]['scores'], predictions[-1]['labels'], predictions[0]['boxes']):
#         if score > 0.5:  # Adjust confidence threshold as needed
#             box = [int(i) for i in box]
#             draw.rectangle(box, outline="red")
#             class_name = COCO_INSTANCE_CATEGORY_NAMES[label]
#             draw.text((box[0], box[1]), f"Class: {class_name}, Score: {score:.2f}", fill="red")
#
#     # Convert the image back to OpenCV format for display
#     result_frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
#
#     # Display the frame
#     cv2.imshow("Object Detection", result_frame)
#
#     # Break the loop when 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # Release the webcam and close all windows
# cap.release()
# cv2.destroyAllWindows()

