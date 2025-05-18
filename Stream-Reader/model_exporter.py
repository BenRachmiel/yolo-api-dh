from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("yolo12n.pt")
    model.export(format='torchscript')
