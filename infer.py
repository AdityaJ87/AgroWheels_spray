# edge_infer.py
import time, json, argparse
import cv2
import torch
import torchvision.transforms as T
import paho.mqtt.client as mqtt
from PIL import Image
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--camera', type=int, default=0)
parser.add_argument('--model', type=str, default='model_traced.pt')
parser.add_argument('--mqtt_broker', type=str, default='192.168.1.100')
parser.add_argument('--plant_id', type=str, default='plant_001')
parser.add_argument('--threshold', type=float, default=0.6)  # example
args = parser.parse_args()

# Load model
model = torch.jit.load(args.model, map_location='cpu')
model.eval()

transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# MQTT setup
client = mqtt.Client()
client.connect(args.mqtt_broker, 1883, 60)
client.loop_start()

# Camera
cap = cv2.VideoCapture(args.camera)
if not cap.isOpened():
    raise SystemExit("Camera not accessible")

def predict(image_bgr):
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img)
    x = transform(pil).unsqueeze(0)
    with torch.no_grad():
        out = model(x)
        probs = torch.nn.functional.softmax(out[0], dim=0).numpy()
    return probs  # vector of probabilities

# You must align classes with training . Use same order when mapping 'disease' classes.
# Example mapping loaded from a JSON file; for demo set:
classes = ['healthy','diseaseA','diseaseB']

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.2)
            continue

        probs = predict(frame)
        top_idx = int(probs.argmax())
        top_prob = float(probs[top_idx])
        label = classes[top_idx]

        print(f"Detected {label} ({top_prob:.2f})")
        # Decide spray logic
        if label != 'healthy' and top_prob > args.threshold:
            # Simple proportional dosage by confidence
            base_ml = 10.0
            dosage = base_ml * (top_prob)
            # Publish command
            command = {
                "plant_id": args.plant_id,
                "label": label,
                "confidence": round(top_prob,3),
                "dosage_ml": round(float(dosage),2),
                "timestamp": int(time.time())
            }
            client.publish('farm/sprayer/command', json.dumps(command), qos=1)
            print("Published spray command:", command)
        # Optionally sleep between captures to avoid double-spray
        time.sleep(3)
except KeyboardInterrupt:
    pass
finally:
    cap.release()
    client.loop_stop()
    client.disconnect()
