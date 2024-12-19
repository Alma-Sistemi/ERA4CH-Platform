from samgeo.text_sam import LangSAM
from PIL import Image
#import torch
from fastapi import FastAPI, File, Form, UploadFile
from starlette.responses import Response
from fastapi.responses import JSONResponse
import io
from pydantic import BaseModel
from models.langsamv2 import inference_LangSam, inference_LangSam_for_mobile_app
from models.crack_segmentation.src.inference import crack_segmentation
import json
import os
from typing import Dict

import numpy as np
import uvicorn

app = FastAPI(title="LangSam - image segmentation with text prompt",
              description='''Obtain semantic segmentation of a desired object in the image via LangSam.
                           Visit this URL at port 8501 for the streamlit interface.''',
              version="0.1.0",)


@app.post("/segmentation")
async def get_segmentation(file: bytes = File(...), text: str = Form(...)):
    print("in fastapi server- post request")

    # print(text)
    masks, boxes, phrases, logits, predictions = inference_LangSam(file, text)
    # Create a response dictionary
    response_data = {
        "masks": masks.tolist(),
        "boxes": boxes.tolist(),
        "phrases": phrases,
        "logits": logits.tolist(),
        "predictions": predictions.tolist()
    }

    # Serialize the response dictionary to JSON
    json_response = json.dumps(response_data)

    return JSONResponse(content=json_response)


@app.post("/cracks-segmentation")
async def get_cracks_segmentation(file: UploadFile = File(...)):
    print("in fastapi server- crack post request")
    file_location = f"uploaded_images/{file.filename}.jpg"
    with open(file_location, "wb+") as file_object:
        file_object.write(file.file.read())
    print(f"file '{file.filename}' saved at '{file_location}'")
    path_to_overlay_image = crack_segmentation(
        "uploaded_images/", out_return=False)

    return Response(content=str(path_to_overlay_image))


if __name__ == '__main__':
	uvicorn.run(app, port=8000, host="0.0.0.0")