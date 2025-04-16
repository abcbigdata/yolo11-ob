from ultralytics import YOLO
import  json
import os
if __name__ == '__main__':
    # Load a model
    #model = YOLO("best_coco8.pt")

    #model = YOLO("best_coco.pt")
    #model = YOLO("/root/PycharmProjects/PythonProject/main/model/apple/weights/best.engine",task="detect")
    model = YOLO("/root/PycharmProjects/PythonProject/main/runs/detect/train/weights/best.engine", task="detect")
    # Train the model

    # Perform object detection on an image
    #results = model("/root/PycharmProjects/PythonProject/main/datasets/test/coco/")
    results = model("/home/data/yolo11/datasets/apple/test/tu1/p2*",stream=False)
    #
    for result in results:

        result.show()

        jsons = json.loads(result.to_json(False,2))
        if len(jsons) == 0 :
            print(result.path, "no detect")

        # for rs in jsons:
        #     print(os.path.basename(result.path), "detect", rs["name"], "百分比：", rs["confidence"])









    # Export the model to ONNX format
    #path = model.export(format="onnx")  # return path to exported model