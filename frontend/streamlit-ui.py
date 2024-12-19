import streamlit as st
import streamlit.components.v1 as components
from requests_toolbelt.multipart.encoder import MultipartEncoder
import requests
from PIL import Image
import io
import json
from utils.utils import apiPostRequest, apiPostRequestForCracks, create_json, show_anns, select_masked_above_threshold, create_mask_overlay,get_street_view_images,closest_panos,calculate_distances
import torch
import numpy as np
from streamlit_option_menu import option_menu
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from json.decoder import JSONDecodeError
import cv2 as cv
from ultralytics import YOLO
import os

from ipywidgets import embed #ipywidgets version 7.7.1 is required
import copy
import supervision as sv
from models.YoloWorldEfficientSam.yoloWorld_EfficientSam import yolo_world_inference,efficientSam_YoloWorld,video_yolo_world_inference,video_efficientSam_yolo_world_inference
from models.vqa_material_age import vision_gpt_inference, inference_vqa_model_for_material,inference_vqa_model_for_age
from models.Floor_detection.Floor_detection import inference_floor_detection
from components.building_inspection_assistant_component import building_inspection_component
from components.heatmaps_from_streetview_component import render_heatmaps_workflow
from models.mesh_visualization_streamlit import mesh_generation
from components.facade_element_detection import facade_element_detection

st.set_page_config(layout="wide")

if "button_clicked" not in st.session_state:
    st.session_state.button_clicked = False

if "element_detected" not in st.session_state:
    st.session_state.element_detected = ""

if "image_name" not in st.session_state:
    st.session_state.image_name = ""


############################################################# ERA4CH PLATFORM ################################################################

st.markdown("<h1 style='text-align: center; color: white;'>ERA4CH Platform</h1>", unsafe_allow_html=True)
# st.markdown('#')
with st.sidebar:
    selected = option_menu("Option Menu", ["Building Inspection Assistant",'Facade Element detection','Damage Assessment Heatmaps'],
                            icons=['globe-americas','eye'],
                             key='menu_5', menu_icon="list", default_index=0)
    st.markdown( 
            """
                <link rel="stylesheet" href="cdn.jsdelivr.net/npm/bootstrap-icons@1.10.2/font/…">
                <div style="bottom:-560px; position:absolute; display:hidden;">
                    <!-- 
                    <h3 style="margin-top:0; text-align:center;">eUMap<h3>
                    -->
                    <img src="https://i.imgur.com/wEelXd7.jpg" width="90%" style="display: block; margin-left:auto; margin-right: auto; position:relative;"/>
                    <p style="margin-top:50px; font-size:80%">
                    Marie Skłodowska-Curie Actions (MSCA)  Research and Innovation Staff Exchange (RISE) H2020-MSCA-RISE-2020 G.A. 101007638
                    </p>
                    <!-- 
                    <img src="https://i.imgur.com/yEF6GB3.png" width="100%" style="bottom:0px; position:relative;"/>
                    
                    
                </div>
            """
        ,unsafe_allow_html=True)

if selected == 'Facade Element detection':
    facade_element_detection()

elif selected == 'Building Inspection Assistant':
    building_inspection_component()

elif selected == 'Damage Assessment Heatmaps':
    render_heatmaps_workflow()