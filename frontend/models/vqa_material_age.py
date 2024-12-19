from transformers import ViltProcessor, ViltForQuestionAnswering,AutoModel, AutoTokenizer
from PIL import Image
import os
import base64
import requests


def inference_vqa_model_for_material(image):

    file_name = image.name
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    file_path = os.path.join("uploads", file_name)  # Define your desired directory
    with open(file_path, "wb") as f:
        f.write(image.getvalue())
    SOURCE_IMAGE_PATH = "./uploads/" + file_name
    image = Image.open(SOURCE_IMAGE_PATH)
    # image = Image.open(requests.get(url, stream=True).raw)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    text = "what is the construction material?"
    print("image shape ", image.size)
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

    # prepare inputs
    encoding = processor(image, text, return_tensors="pt")

    # forward pass
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    print("Predicted answer:", model.config.id2label[idx])
    return model.config.id2label[idx]

def inference_vqa_model_for_age(image):

    file_name = image.name
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    file_path = os.path.join("uploads", file_name)  # Define your desired directory
    with open(file_path, "wb") as f:
        f.write(image.getvalue())
    SOURCE_IMAGE_PATH = "./uploads/" + file_name
    image = Image.open(SOURCE_IMAGE_PATH)
    # image = Image.open(requests.get(url, stream=True).raw)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    text = "What is the period of construction of the building? your answer can be: 1800-1900 or 1900-2000 or 2000-2024]"
    print("image shape ", image.size)
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")

    # prepare inputs
    encoding = processor(image, text, return_tensors="pt")

    # forward pass
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    print("Predicted answer:", model.config.id2label[idx])
    return model.config.id2label[idx]




def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
def vision_gpt_inference(image):

    file_name = image.name
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    file_path = os.path.join("uploads", file_name)  # Define your desired directory
    with open(file_path, "wb") as f:
        f.write(image.getvalue())
    image_path = "./uploads/" + file_name

    # OpenAI API Key
    api_key = 'XXXXXXXXXXXXXXXX'

    # Path to your image
    # image_path = "/content/DSC01141-CROP.png"

    # Getting the base64 string
    base64_image = encode_image(image_path)

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
    }

    payload = {
    "model": "gpt-4o",
    "messages": [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": "what is the estimated construction period of the building? Give me a range in dates, like 1800-1850.No yapping"
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
            }
        ]
        }
    ],
    "max_tokens": 300
    }
    # UNCOMMENT THE BELOW LINES TO USE API
    # response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    # return response.json()['choices'][0]['message']['content']
    return '1620-1640'  #for DSC01141-CROP.png


def vision_gpt_inference_building_utilization(image):

    file_name = image.name
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    file_path = os.path.join("uploads", file_name)  # Define your desired directory
    with open(file_path, "wb") as f:
        f.write(image.getvalue())
    image_path = "./uploads/" + file_name

    # OpenAI API Key
    api_key = 'XXXXXXXXXXXXX'

    # Path to your image
    # image_path = "/content/DSC01141-CROP.png"

    # Getting the base64 string
    base64_image = encode_image(image_path)

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
    }

    payload = {
    "model": "gpt-4o",
    "messages": [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": "Can you find the utilization of that building? Pick one one on the following options: Residential,Offices,Shops,\
                    Supermarket,Entertainment, Theater,Hospital,Church,School / University,\
                    Bank,  Public use,  Consulting services, \
                    Emergency service,  Parking,  Care services, \
                    Wholesale shop, Warehouse,   Workshop, \
                    Heavy industry, Light industry. No yapping"
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
            }
        ]
        }
    ],
    "max_tokens": 300
    }
    # UNCOMMENT THE BELOW LINES TO USE API
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    print("The result issssss ", response.json()['choices'][0]['message']['content'])
    return response.json()['choices'][0]['message']['content']
    # return 'School / University' 


def vision_gpt_inference_building_utilization_mobile_app(image):

    file_name = image.name
    if not os.path.exists("uploads"):
        os.makedirs("uploads")
    file_path = os.path.join("uploads", file_name)  # Define your desired directory
    with open(file_path, "wb") as f:
        f.write(image.getvalue())
    image_path = "./uploads/" + file_name

    # OpenAI API Key
    api_key = 'XXXXXXXXXX'

    # Path to your image
    # image_path = "/content/DSC01141-CROP.png"

    # Getting the base64 string
    base64_image = encode_image(image_path)

    headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
    }

    payload = {
    "model": "gpt-4o",
    "messages": [
        {
        "role": "user",
        "content": [
            {
            "type": "text",
            "text": "Can you find the utilization of that building? Pick one one on the following options: Residential,Offices,Shops,\
                    Supermarket,Entertainment, Theater,Hospital,Church,School / University,\
                    Bank,  Public use,  Consulting services, \
                    Emergency service,  Parking,  Care services, \
                    Wholesale shop, Warehouse,   Workshop, \
                    Heavy industry, Light industry. No yapping"
            },
            {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
            }
            }
        ]
        }
    ],
    "max_tokens": 300
    }
    # UNCOMMENT THE BELOW LINES TO USE API
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    print("The result issssss ", response.json()['choices'][0]['message']['content'])
    return response.json()['choices'][0]['message']['content']
    # return 'School / University' 