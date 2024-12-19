from samgeo.text_sam import LangSAM
import io

import torch
from PIL import Image
import json
import os
import shutil


def inference_LangSam(binary_image, text):

    # Initialize LangSAM class. Δownloads the model weights and sets up the model for inference.

    sam = LangSAM()

    # can be parameterised -> user can select the object for detection
    #text_prompt = "window"
    text_prompt = text

    image = Image.open(io.BytesIO(binary_image)).convert("RGB")

    masks, boxes, phrases, logits, predictions = sam.predict(
        image, text_prompt, box_threshold=0.2, text_threshold=0.24, return_results=True)
    print(text + " detection")

    return masks, boxes, phrases, logits, predictions

def inference_LangSam_for_mobile_app(image, text):

    file_name = image.filename
    file_name = file_name.split("/")
    file_name = file_name[-1]
        
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    file_path = os.path.join("uploads", file_name)  # Define your desired directory

    if os.path.exists(file_path):
        print("File exists")
    else:
        print("File does not exist")
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(image.file, buffer)
        finally:
            image.file.close()

    # Initialize LangSAM class. Δownloads the model weights and sets up the model for inference.

    sam = LangSAM()

    # can be parameterised -> user can select the object for detection
    #text_prompt = "window"
    text_prompt = text

    image = Image.open(file_path).convert("RGB")

    masks, boxes, phrases, logits, predictions = sam.predict(
        image, text_prompt, box_threshold=0.3, text_threshold=0.24, return_results=True)
    print(text + " detection")

    return masks, boxes, phrases, logits, predictions