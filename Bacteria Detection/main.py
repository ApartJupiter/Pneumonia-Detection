from ultralytics import YOLO
import torch




# print(torch.cuda.is_available())
# print(torch.__version__)
# print(torch.version.cuda)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f'Using device: {device}')

if __name__ == '__main__':
    model = YOLO('yolov8n.pt')
    results = model.train(data=r'E:\Gurjyot\Navrachna Uni\3rd YEAR\SEM 6\CV LAb\proj disp\tets - 2\set - 1\data.yaml', batch=2, epochs=100, device=0, imgsz=640, close_mosaic=0)


