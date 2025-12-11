from ultralytics import YOLO

def main():
    dataset_yaml = '/home/caio/Área de trabalho/projeto_ML/yolo/data.yaml'
    
    model = YOLO('yolov8n.pt')
    
    model.train(
        data=dataset_yaml,
        epochs=100,
        imgsz=640,
        batch=8,
        device=0,
        name='treino_v1_osteoporose',
        patience=20,
        workers=4,
        exist_ok=True
    )

if __name__ == '__main__':
    main()