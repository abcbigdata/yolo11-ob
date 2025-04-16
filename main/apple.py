from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO("yolo11n.pt")

    # Train the model
    train_results = model.train(
        data="apple.yaml",  # path to dataset YAML
        epochs=130,  # number of training epochs
        imgsz=640,  # training image size
        device="0",  # device to run on, i.e. device=0 or device=0,1,2,3 or device=cpu
    )

    # Evaluate model performance on the validation set
    metrics = model.val()

    # Perform object detection on an image ==model.predict
    # results = model("/root/PycharmProjects/PythonProject/main/datasets/apple/test/")
    # results[0].show()

    # Export the model to ONNX format
    path = model.export(format="engine")  # return path to exported model