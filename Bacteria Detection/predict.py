from ultralytics import YOLO

lst=[0]

# Load a model
model = YOLO(r'E:\Gurjyot\Navrachna Uni\3rd YEAR\SEM 6\CV LAb\proj disp\tets - 2\runs\detect\train 3\weights\best.pt')  # load a custom model


# Predict with the model
results = model(r'E:\Gurjyot\Navrachna Uni\3rd YEAR\SEM 6\CV LAb\proj disp\tets - 2\set - 1\test\images\BACTERIA-2034017-0002_jpeg.rf.f832f1693ca42834976feffa424ccea5.jpg',imgsz=640, classes=lst, save=True)

model2 = YOLO(r'yolov8s.pt')

