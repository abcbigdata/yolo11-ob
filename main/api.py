from fastapi import FastAPI , File, UploadFile
import cv2
import numpy as np
import json,os
from ultralytics import YOLO

app = FastAPI(title="YOLOv11 Detection Service")
model = YOLO("/root/PycharmProjects/PythonProject/main/model/apple/weights/best.engine",task="detect")


@app.post("/detect")
async def detect(image: UploadFile = File(...)):
    # 读取并预处理图像
    img_bytes = await image.read()
    img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    # 推理
    results = model(img, stream=False)  # 禁用流模式提升性能
    for result in results:
        jsons = json.loads(result.to_json(False,2))
        for rs in jsons:
            print(os.path.basename(result.path), "detect", rs["name"], "百分比：", rs["confidence"])
    # 格式化输出
    return results



@app.get("/")
async def init():
    return "hello world"