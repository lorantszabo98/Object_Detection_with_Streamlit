# Object Detection with Streamlit

This Streamlit app performs object detection on uploaded images using various pre-trained models. The app provides insights into the detection results, including the number of detections per label and a model comparison across different images.

# Requirements
- Python
- Streamlit
- Streamlit Extras
- OpenCV
- PyTorch
- Numpy
- Pandas
- Plotly
- AST

# Instructions
Running:
  - Execute `streamlit run your_app_filename\Object_Detection.py` in the terminal.
  - Access the app in your browser at http://localhost:8501.

# Features

Object Detection Page
- Upload and detect objects in images.
- Choose from multiple pre-trained models.
- Adjust the confidence threshold for filtering detections.
- View the original image with bounding boxes and labels.
- Display the number of detections per label.

Object Detection Results Page
- Visualize the distribution of detections per label in a bar chart.
- Compare models across different images.
- Display detection statistics and images with detected objects.
- View the original image alongside detection results.

# Example Usage

- Open the Streamlit app using the provided installation instructions.
- Upload an image on the Object Detection Page.
- Choose a detection model and set the confidence threshold.
- Click the "Perform object detection" button.
- Explore the results on the Object Detection Results Page.
- Feel free to experiment with different images, models, and settings to analyze object detection performance.
