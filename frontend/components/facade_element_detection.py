import streamlit as st
import streamlit.components.v1 as components
from requests_toolbelt.multipart.encoder import MultipartEncoder
from PIL import Image
from utils.utils import apiPostRequest, apiPostRequestForCracks,show_anns,create_json,select_masked_above_threshold, create_mask_overlay,get_street_view_images,closest_panos,calculate_distances
from PIL import Image
import io
import numpy as np
import json
from streamlit_option_menu import option_menu
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from json.decoder import JSONDecodeError
import cv2 as cv
from ultralytics import YOLO
import os

from ipywidgets import embed #ipywidgets version 7.7.1 is required
import supervision as sv
from models.YoloWorldEfficientSam.yoloWorld_EfficientSam import video_yolo_world_inference,video_efficientSam_yolo_world_inference
from models.vqa_material_age import vision_gpt_inference, inference_vqa_model_for_material,inference_vqa_model_for_age
from models.Floor_detection.Floor_detection import inference_floor_detection

from models.mesh_visualization_streamlit import mesh_generation

def callback(elem_for_det,image_name):
    st.session_state.button_clicked = True
    st.session_state.element_detected = elem_for_det
    st.session_state.image_name = image_name
    print("In callback the elem is, ", elem_for_det)
    print("In callback the image name is, ", image_name)
# It is not hashable/cachable , because of predictions
def lagSAM_predictions_process(predictions,image):
    # Attempt to parse the JSON data
    data = predictions.json()
    dict_data = json.loads(data)
    logits_list = dict_data["logits"]
    image_pil = Image.open(io.BytesIO(image.getvalue()))
    print("In cached process predictions")
    return dict_data,logits_list,image_pil

@st.cache_data
def first_api_request(image,url,element_for_detection):
    predictions = apiPostRequest(image, url, element_for_detection)
    print("In cached first api request")
    print("name of image ",image.name)
    dict_data,logits_list,image_pil = lagSAM_predictions_process(predictions,image)
    return predictions,dict_data,logits_list,image_pil


# It is not hashable/cachable , because of image_pil
def render_segmented_image(element_for_detection,image_pil,dict_data, logits_list, threshold_from_slider):

    #threshold_from_slider = 0.3
    keep_boxes_tensor, keep_masks_tensor, keep_logits_list = select_masked_above_threshold(
                            dict_data["boxes"], dict_data["masks"], logits_list, threshold_from_slider)

    predictions_tensor_above_threshold = create_mask_overlay(
        image_pil, keep_boxes_tensor, keep_masks_tensor)

    json_string= None
    
    if predictions_tensor_above_threshold is None:
        st.write("No objects with confidence score >",
                threshold_from_slider, " found in the image. Try lower threshold")
    else:
        show_anns(
            image_pil,
            predictions_tensor_above_threshold,
            keep_boxes_tensor,
            detected_element=element_for_detection,
            cmap='Reds',
            add_boxes=True,
            alpha=0.5,
            output="segmented.png"
            # figsize=(12, 10)
        )
        print("after show anns")

        json_string = create_json(
            keep_logits_list, keep_boxes_tensor, keep_masks_tensor, element_for_detection)

        segmented_image = Image.open(r"segmented.png")
        st.image(segmented_image, width=700,
        caption="Segmented image",use_column_width=True)

    return json_string,predictions_tensor_above_threshold

def reset_run_model_button(element_for_detection,image_name):

    # Detect if the detected element or image, has been changed after the button was pressed
    if ((st.session_state.element_detected != element_for_detection) or (st.session_state.image_name != image_name)) and st.session_state.button_clicked:
        st.session_state.button_clicked = False


def facade_element_detection():

    # fastapi endpoint
    url = 'http://127.0.0.1:8000'
    endpoint = '/segmentation'
    st.header('Facade Element Detection', divider='gray')
    _, col, _ = st.columns([0.25, 0.5, 0.25])

    with col:

        st.markdown("<h5 style='text-align: center; color: white;'>In this workflow you can perform facade identification using State of the Art computer vision models. Upload your image/video and choose the element of your interest. A report with all the extracted information is generated and is available to download</h5>", unsafe_allow_html=True)

        image = st.file_uploader('Upload a facade image') 
        
        if image != None and image.type  != "video/mp4":
            
            el_col1,_,det_col2 = st.columns([0.2,0.6,0.2])
            st.markdown('#')
            with _:
                element_for_detection = st.radio(
                    "Choose the element that you want to detect:",
                    ["Window", "Door", "Balcony", "Crack","Floor","Material","Age", "Other damages","Mesh Generation"],horizontal=True,
                    index=None,
                )
            # with det_col2:
            #     detection_technique = st.radio(
            #         "Choose the detection technique of your interest:",
            #         ["Object detection", "Segmentation"],
            #         index=1,
            #     )

            if element_for_detection != None:

                reset_run_model_button(element_for_detection,image.name)
                col11, col22, col33 = st.columns(3)
                with col22:
                    segmentation_button = st.button(
                        'Run model', on_click=callback, args=[element_for_detection,image.name],use_container_width=True)

                # the below code is executed for the cracks detection
                if segmentation_button  and element_for_detection == "Crack":
                    endpoint_for_cracks = "/cracks-segmentation"
                    path_to_overlay_image = apiPostRequestForCracks(
                        image, url+endpoint_for_cracks)
                    segmented_image = Image.open(path_to_overlay_image)
                    st.image(segmented_image, width=700, caption="Segmented image")

                # the below code is executed for the material identification
                elif segmentation_button and element_for_detection == "Material":
                    detected_material = inference_vqa_model_for_material(image)
                    material_image_caption = "The construction material of the facade is: " + detected_material
                    st.image(image,use_column_width=True)
                    markdown_text = f"<p style='text-align: center ;font-size: 20px; color: white;'>{material_image_caption} </p>"
                    st.markdown(markdown_text, unsafe_allow_html=True)

                elif segmentation_button and element_for_detection == "Age":
                    detected_age = vision_gpt_inference(image)
                    age_image_caption = "The estimated age of the building is: " + detected_age
                    st.image(image,use_column_width=True)
                    markdown_text = f"<p style='text-align: center ;font-size: 20px; color: white;'>{age_image_caption} </p>"
                    st.markdown(markdown_text, unsafe_allow_html=True)

                elif segmentation_button and element_for_detection == "Other damages":

                    
                    model = YOLO('C:/Users/napol/Desktop/Alma-Sistemi/ERA4CH/Koutmos Thesis/ModelPredictionDipl/best100.pt')
                    file_name = image.name
                    if not os.path.exists("uploads"):
                        os.makedirs("uploads")
                    file_path = os.path.join("uploads", file_name)  # Define your desired directory
                    with open(file_path, "wb") as f:
                        f.write(image.getvalue())
                    SOURCE_IMAGE_PATH = "./uploads/" + file_name
                    image = cv.imread(SOURCE_IMAGE_PATH)
                    results = model.predict(image, save=False)
                    img = results[0].plot(font_size=20, pil=True)
                    st.image(img,use_column_width=True)

                elif segmentation_button and element_for_detection == "Mesh Generation":
                    mesh_generation(image)

                elif segmentation_button and element_for_detection == "Floor":
                    floor_annotated_image,num_of_floors = inference_floor_detection(image)
                    st.image(floor_annotated_image,use_column_width=True)
                    floor_image_caption = "The number of floors is " + str(num_of_floors)
                    markdown_text = f"<p style='text-align: center ;font-size: 20px; color: white;'>{floor_image_caption} </p>"
                    st.markdown(markdown_text, unsafe_allow_html=True)


                elif (segmentation_button or (segmentation_button==False and st.session_state.button_clicked)) and (element_for_detection not in ["Crack","Material","Floor"]) :
                    
                    # This try-catch is handling the case in which no object is found in the photo.
                    try:
                        predictions,dict_data,logits_list,image_pil = first_api_request(image, url+endpoint, element_for_detection)
                        threshold_from_slider = st.slider(
                            'Choose the threshold of the predictions that you want to visualise', 0.2, 1.0, value=0.2, step=0.05)
                        json_string,predictions_tensor_above_threshold = render_segmented_image(element_for_detection,image_pil,dict_data, logits_list, threshold_from_slider)
                        
                        if predictions_tensor_above_threshold is not None:
                            col111, col222, col333 = st.columns([0.4, 0.2, 0.4])
                            with col111:
                                st.download_button(
                                    label="Download elements' information",
                                    file_name="information.json",
                                    mime="application/json",
                                    data=json_string,
                                )
                        

                    except JSONDecodeError as e:
                        # Handle the JSONDecodeError and provide feedback to the user
                        st.write("No objects found in the image. Try another Element!")

        # Video segmentation/object detection
        elif image != None and image.type == "video/mp4":
            video = image
            el_col1,_,det_col2 = st.columns([0.4,0.2,0.4])
            with el_col1:
                element_for_detection_video = st.radio(
                    "Choose the element that you want to detect",
                    ["Window", "Door", "Balcony"],
                    index=0,
                )
            with det_col2:
                detection_technique = st.radio(
                    "Choose the detection technique of your interest:",
                    ["Object detection", "Segmentation"],
                    index=0,
                )
            st.markdown('#')
            col1, col2, col3 = st.columns(3)
            with col2:
                segmentation_button = st.button(
                    'Run model',use_container_width=True)
            if segmentation_button and detection_technique == "Object detection":
                video_yolo_world_inference(video,element_for_detection_video)

                st.success('Your video has been processed and saved successfully!', icon="✅")
            elif segmentation_button and detection_technique == "Segmentation":
                video_efficientSam_yolo_world_inference(video,element_for_detection_video)
                st.success('Your video has been processed and saved successfully!', icon="✅")