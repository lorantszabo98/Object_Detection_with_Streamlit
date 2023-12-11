import torch
from torchvision.models import detection
import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageOps
import pandas as pd
import time


@st.cache_resource(show_spinner="Loading the model...")
def load_models(model_name):
    model = MODELS[model_name](pretrained=True, progress=True, num_classes=len(CLASSES), pretrained_backbone=True).to(DEVICE)
    # set inference mode
    model.eval()

    return model


@st.cache_data
def load_coco_labels(filename):
    with open(filename, 'r') as file:
        lines = [line.strip() for line in file.readlines()]

    return lines


CLASSES = load_coco_labels("pages/data/coco_labels.txt")

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODELS = {
    "frcnn-resnet": detection.fasterrcnn_resnet50_fpn,
    "frcnn-mobilenet": detection.fasterrcnn_mobilenet_v3_large_320_fpn,
    "retinanet": detection.retinanet_resnet50_fpn
    # you can add more models, if you want
}

# Create a session state to store the DataFrame
if 'detection_df' not in st.session_state:
    st.session_state.detection_df = pd.DataFrame()

if 'full_analysis' not in st.session_state:
    st.session_state.full_analysis = pd.DataFrame()

st.title("Object Detection with Streamlit")

st.info("To perform object detection, please upload a photo!")

image = st.file_uploader("Please upload a photo!", accept_multiple_files=False, type=["jpg", "jpeg", "png"])

image_placeholder = st.empty()

if image:

    # dynamically update the selectbox from the MODELS dictionary
    model_names = list(MODELS.keys())

    selected_model = add_selectbox = st.selectbox(
        "Which model do you want to use for detection?",
        model_names,
        index=0
    )

    confidence_threshold = st.slider("Please set the confidence threshold value (%)", 0, 100, 50)
    confidence_threshold = confidence_threshold/100

    analysis_results = {'Image': image.name, 'Model': [], 'Detection time': [],
                        'Confidence threshold': confidence_threshold, 'Labels_Confidence': []}

    if st.button("Perform object detection"):

        model = load_models(selected_model)

        analysis_results['Model'] = selected_model

        with st.spinner("Object detection in progress..."):

            start_time = time.time()

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

            data_rows = []

            # loop over the detections
            for i in range(0, len(detections["boxes"])):
                # extract the confidence (i.e., probability) associated with the
                # prediction
                confidence = detections["scores"][i]
                # filter out weak detections by ensuring the confidence is
                # greater than the minimum confidence
                if confidence > confidence_threshold:
                    # extract the index of the class label from the detections,
                    # then compute the (x, y)-coordinates of the bounding box
                    # for the object
                    idx = int(detections["labels"][i])
                    box = detections["boxes"][i].detach().cpu().numpy()
                    (startX, startY, endX, endY) = box.astype("int")
                    # display the prediction to our terminal
                    label = "{}: {:.2f}%".format(CLASSES[idx-1], confidence * 100)
                    # st.write(label)
                    # draw the bounding box and label on the image
                    cv2.rectangle(orig, (startX, startY), (endX, endY),
                                  COLORS[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    # sgow the label in the photo
                    cv2.putText(orig, label, (startX, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLORS[idx], 4)

                    # create dictionary to store label and confidence values
                    detection_info = {
                        'Label': CLASSES[idx-1],
                        'Confidence': "{:.2f}%".format(float(confidence * 100)),
                        # Add more fields as needed
                    }

                    # add dictionary to a list
                    data_rows.append(detection_info)

            # measure the detection time
            end_time = time.time()
            elapsed_time = end_time - start_time
            st.toast(f"Object detection took {elapsed_time:.4f} seconds")

            analysis_results['Detection time'] = elapsed_time

        # show the image with detecton results
        image_placeholder = st.image(orig, channels="RGB", use_column_width=True)

        # show the detection results in a dataframe
        df = pd.DataFrame(data_rows)
        st.dataframe(df)

        # df.to_csv("pages/data/data.csv", mode='a', header=False, index=False)

        # add this dataframe to the Session state to perform further processing in the other page
        st.session_state.detection_df = df

        analysis_results['Labels_Confidence'] = data_rows


        full_analysis_df = pd.DataFrame(analysis_results)

        consolidated_df = full_analysis_df.groupby(['Image', 'Model', 'Detection time', 'Confidence threshold']).agg(lambda x: ', '.join(x.astype(str))).reset_index()
        st.dataframe(consolidated_df)

        consolidated_df.to_csv("pages/data/data_full.csv", mode='a', header=False, index=False)


