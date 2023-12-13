import streamlit as st
import pandas as pd
import ast
import plotly.express as px
from streamlit_extras.grid import grid
import os


def load_data(filepath):
    df = pd.read_csv(filepath)
    return df


def save_data(filename, new_dataframe):
    new_dataframe.to_csv(filename, index=False)


st.title("Object Detection Results ")
tab1, tab2 = st.tabs(['Number of detections', 'Model comparison'])

# # load the dataframe
loaded_dataframe = load_data("pages/data/data.csv")

# get the detection results as a dataframe from the Session state
if "detection_df" not in st.session_state:
    st.session_state.detection_df = []
# Retrieve the DataFrame from session state
detection_df = st.session_state.detection_df
detection_df = pd.DataFrame(detection_df)

if not detection_df.empty:
    for index, row in detection_df.iterrows():
        label = row['Label']

        # Check if the label is already present in loaded_dataframe
        if label in loaded_dataframe['Label'].values:
            # If present, increment the 'Number of Detections'
            loaded_dataframe.loc[loaded_dataframe['Label'] == label, 'Number of Detections'] += 1
        else:
            # If not present, add a new row to loaded_dataframe
            new_row = {'Label': label, 'Number of Detections': 1}
            loaded_dataframe = loaded_dataframe.append(new_row, ignore_index=True)

# Sort the DataFrame based on the 'Number of Detections' column
loaded_dataframe = loaded_dataframe.sort_values(by='Number of Detections', ascending=False)

# Plot the bar chart using Plotly Express
fig = px.bar(loaded_dataframe, x='Label', y='Number of Detections',
             color='Label',
             labels={'Number of Detections': 'Number of Detections'},
             title='Number of detections of each label',
             template='plotly',
             height=500, )

with tab1:
    st.info("On this tab you can see all detections per label summarised.")
    st.dataframe(loaded_dataframe, use_container_width=True, hide_index=True)

    # Display the chart
    st.plotly_chart(fig)

# clear the Session state
st.session_state.detection_df = []

save_data("pages/data/data.csv", loaded_dataframe)

########################


loaded_dataframe_full = load_data("pages/data/data_full.csv")

base_path = "http://localhost:8501/app/static/"
loaded_dataframe_full['Displayed Image'] = loaded_dataframe_full['Image'].apply(lambda x: os.path.join(base_path, x))

loaded_dataframe_full.insert(0, 'Displayed Image', loaded_dataframe_full.pop('Displayed Image'))

with tab2:
    st.info("On this tab you can see how many objects each model found in the detection, image by image.")
    st.dataframe(
        loaded_dataframe_full,
        column_config={"Displayed Image": st.column_config.ImageColumn(
            "Displayed Image",
            help="The images in which the object detection took place"
        )
        },
        use_container_width=True,
        hide_index=True)

# Convert the string representations of lists into actual lists of dictionaries
loaded_dataframe_full['Labels_Confidence'] = loaded_dataframe_full['Labels_Confidence'].apply(ast.literal_eval)

# If the confidence threshold is not the default (which is 0.5), then this detection is not considered in the model comparison
loaded_dataframe_full = loaded_dataframe_full[loaded_dataframe_full['Confidence threshold'] == 0.5]

# Create a list to store data for plotting
data_for_plotting = []

# Drop non-unique rows based on 'Image' and 'Model'
unique_rows = loaded_dataframe_full.drop_duplicates(subset=['Image', 'Model'])

# Iterate over the unique rows and collect all labels for each unique picture and model pair
for index, row in unique_rows.iterrows():
    all_labels = []
    labels_confidence = row['Labels_Confidence']

    # Check if Labels_Confidence is a dictionary
    if isinstance(labels_confidence, dict):
        label = labels_confidence.get('Label')  # Use get() method to avoid KeyError
        if label is not None:
            all_labels.append(label)
    else:
        for entry in labels_confidence:
            if isinstance(entry, dict):
                label = entry.get('Label')
                if label is not None:
                    all_labels.append(label)

    data_for_plotting.append({
        'Image': row['Image'],
        'Model': row['Model'],
        'Total Labels': len(all_labels),
        'Detection Time': row['Detection time']
    })

# Create a DataFrame from the collected data
plotting_df = pd.DataFrame(data_for_plotting)

# Plot the bar chart
fig = px.bar(plotting_df, x='Image', y='Total Labels', title="Model comparison", color='Model', barmode='group',
             hover_data=['Detection Time'], labels={'Detection Time': 'Detection Time'})

with tab2:
    # Display the chart
    st.plotly_chart(fig)
