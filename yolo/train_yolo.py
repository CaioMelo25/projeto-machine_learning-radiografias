import os
from ultralytics import YOLO

def main():
    pasta_atual = os.path.dirname(os.path.abspath(__file__))
    
    dataset_yaml = os.path.join(pasta_atual, 'data.yaml')
    
    print(f"Usando configuração em: {dataset_yaml}")
    
    
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