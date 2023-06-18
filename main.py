# import cv2
# import numpy as np
# import tensorflow as tf

# from fastapi import FastAPI, UploadFile, File, Request,Form
# from fastapi.templating import Jinja2Templates
# from fastapi.staticfiles import StaticFiles
# from pydantic import BaseModel
# from typing import Annotated
# import h5py
# from PIL import Image

import uvicorn
from fastapi import FastAPI

from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse

import os


app = FastAPI()
app.mount("/static", StaticFiles(directory = "static"), name = "static")


# @app.get("/")
# async def root():
#    return {"message": "hello world"}

@app.get("/")
def read_root():
    with open("templates/base.html", 'r') as file:
        content = file.read()
    return HTMLResponse(content=content)
from fastapi.responses import FileResponse

# @app.get("/video")
# async def get_video():
#     return FileResponse("static/boxvedio.mp4", media_type="video/mp4")
@app.get("/image")
async def get_image():
    return FileResponse("static/images/Background.webp")

# To run locally
if __name__ == '__main__':
   uvicorn.run(app, host='0.0.0.0', port=8000)

# app = FastAPI()
# app.mount("/static", StaticFiles(directory="static"), name="static")

# templates = Jinja2Templates(directory="templates")
# @app.get("/")
# async def dynamic_file(request: Request):
#     return templates.TemplateResponse("base.html", {"request": request})


# def predict_single_img(img_path):
#     pic=[]
#     img = cv2.imread(str(img_path))
#     img = cv2.resize(img, (28,28))
#     if img.shape[2] ==1:
#         img = np.dstack([img, img, img])
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#     img=np.array(img)
#     img = img/255
#     #label = to_categorical(0, num_classes=2)
#     pic.append(img)
#     pic1 = np.array(pic)
#     savedmodel= tf.keras.models.load_model("model.h5")
#     a=savedmodel.predict(pic1)
#     print(a.argmax())
# predict_single_img("img.jpg")

# @app.post("/upload")
# async def upload_image(image: UploadFile = Form(...), patient_details: dict = Form(...)):
#     # Predict probabilities using your model
#     contents = await image.read()
#     img = Image.open(io.BytesIO(contents))
#     return predict_single_img(img)