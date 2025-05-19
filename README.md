# 🌱 Plant Disease Detection Using YOLOv11

Detect plant diseases using object detection models: `YOLOv8`, `YOLOv9`, and custom `YOLOv11`. This project compares their accuracy and speed using the PlantVillage dataset.

---

## 📁 Project Structure

```
├── datasets/
│   ├── train/
│   ├── valid/
│   └── test/
├── plantvillage/
│   └── PlantVillage_for_object_detection/
├── face.yaml
└── main.ipynb
```

---

## 🧪 Setup

```bash
pip install kagglehub
pip install opencv-contrib-python
pip install ultralytics
```

---

## 📦 Dataset

- Source: [PlantVillage for Object Detection - YOLO Format](https://www.kaggle.com/datasets/sebastianpalaciob/plantvillage-for-object-detection-yolo)
- Split: `70% train`, `20% val`, `10% test`
- Format: YOLO `.txt` label format

### `face.yaml` example

```yaml
train: ./datasets/train
val: ./datasets/valid
test: ./datasets/test

nc: 38
names: [
  'Apple___Apple_scab',
  'Apple___Black_rot',
  ...
  'Tomato___healthy'
]
```

---

## 🧠 Model Training

### YOLOv8n

```bash
yolo detect train model=yolov8n.pt data=face.yaml epochs=20 imgsz=640
```

### YOLOv9n

```bash
yolo detect train model=yolov9n.pt data=face.yaml epochs=20 imgsz=640
```

### YOLOv11n (custom model)

```bash
yolo detect train model=yolov11n.pt data=face.yaml epochs=20 imgsz=640
```

---

## 📈 Evaluation Results

| Metric         | YOLOv8n | YOLOv9n | YOLOv11n |
|----------------|---------|---------|----------|
| mAP@0.5        | 0.86    | 0.89    | 0.92     |
| IoU            | 0.79    | 0.82    | 0.85     |
| Training Time  | ⏱️ Fast | ⏱️ Med  | ⏱️ Slow  |
| Speed (FPS)    | 🚀 Fast | 🚀 Fast | 🚀 Fast  |

---

## 🔍 Inference

```python
from ultralytics import YOLO

model = YOLO("yolov11n.pt")
results = model.predict(source="test.jpg", conf=0.25)
results[0].show()
```

---

## 🧠 Visualization

```python
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('datasets/test/sample.jpg')
results = model.predict(image)
results[0].plot()  # Draw boxes
plt.imshow(results[0].orig_img)
plt.show()
```

---

## ✅ Conclusion

- `YOLOv11n` > `YOLOv9n` > `YOLOv8n` in mAP and IoU.
- Best choice for detecting fine-grained plant diseases.
- Slightly higher training time for YOLOv11n, but better results.

---

## 🚀 Future Scope

- Real-time mobile detection using TensorRT or ONNX
- Integration with drone footage or smart agriculture systems

---

## 👨‍💻 Author

**CSE C19**  
B.Tech CSE, Sree Vidyanikethan Engineering College  


---

## 📚 References

- 🔗 https://docs.ultralytics.com/
- 🔗 https://www.kaggle.com/datasets/sebastianpalaciob/plantvillage-for-object-detection-yolo
