# coding=utf-8
"""
simple-api to serve.

@copyright  lemoncloud.io 2020
"""
import os
import json
import subprocess
import numpy as np

from PIL import Image
from io import BytesIO
from pydantic import BaseModel
from starlette.responses import FileResponse
from fastapi import FastAPI, File, HTTPException
from starlette.middleware.cors import CORSMiddleware
from models import infer_image, as_path

app: FastAPI = FastAPI(
    title="Shoes Object Detector",
    description='powered by lemoncloud.io',
    docs_url="/",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Custom_Error(BaseModel):
    success: str
    message: str

@app.get(
    "/info",
    responses={400: {"model": Custom_Error}},
    summary="Get Info",
    tags=["Get Info"],
)
def get_info() -> dict:
    """Returns the infor"""
    info = infer_image('./data/shoes.jpg', './result.png')
    result: dict = {"success": True, "start_time": '', "data": info}
    return result

@app.get(
    "/out/{name}",
    summary="Show Image",
    tags=["Show Image"],
)
def get_image_out(name):
    out = as_path('./data/O{}.png'.format(name), read=True)
    return FileResponse(out, media_type="image/png")

@app.post(
    "/predict",
    summary="Upload image and get its predictions using last saved weights",
    tags=["Inference"],
)
async def get_prediction(
    image: bytes = File(..., description="Image to perform inference on")
):
    """Runs the last saved weights to infer on the given image"""
    import time
    current_milli_time = lambda: int(round(time.time() * 1000))
    name = current_milli_time()
    inf = as_path('./data/I{}.jpg'.format(name), read=False)
    out = as_path('./data/O{}.png'.format(name), read=False)
    ret = None
    try:
        img: Image = Image.open(BytesIO(image)).convert("RGB")
        img.save(inf)
        ret = infer_image(inf, out)
        if ret: ret['name'] = name
    except Exception as ex:
        raise HTTPException(422, detail="Error: {}".format(str(ex)))
    # return FileResponse(out, media_type="image/png")
    return ret
