from ultralytics import YOLO
import os

def main():
    pasta_atual = os.path.dirname(os.path.abspath(__file__))
    dataset_yaml = os.path.join(pasta_atual, 'data.yaml')
    
  
    model = YOLO('yolov8s.pt')

    model.train(
        data=dataset_yaml,
        epochs=100,        
        imgsz=640,
        batch=4,           
        device=0,
        name='treino_v2_small_rect',
        patience=30,       
        workers=4,
        
   
        rect=True,        
        cos_lr=True,       
        close_mosaic=10,   
        
       
        degrees=5.0,       
        scale=0.5,         
        fliplr=0.5,        
        mosaic=1.0,        
    )

if __name__ == '__main__':
    main()