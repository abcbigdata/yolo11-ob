from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO("model/coco8/best_coco8.pt")

    # Export the model to ONNX format
    path = model.export(format="engine")  # return path to exported model