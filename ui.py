# ui_app.py
import streamlit as st
import paho.mqtt.client as mqtt
import base64, json, time
from pathlib import Path

MQTT_BROKER = "192.168.1.100"
topic_cmd = "farm/sprayer/command"

client = mqtt.Client()
client.connect(MQTT_BROKER, 1883, 60)
client.loop_start()

st.title("Smart Sprayer Dashboard")
plant_id = st.text_input("Plant ID", value="plant_001")

col1, col2 = st.columns(2)
with col1:
    if st.button("Capture & Analyze (simulate)"):
        # In a real setup, trigger edge to capture & upload; here we just simulate a command
        cmd = {"plant_id": plant_id, "label": "diseaseA", "confidence": 0.85, "dosage_ml": 8, "timestamp": int(time.time())}
        client.publish(topic_cmd, json.dumps(cmd), qos=1)
        st.success("Published spray command: " + str(cmd))
with col2:
    if st.button("Manual Spray 5 ml"):
        cmd = {"plant_id": plant_id, "label": "manual", "confidence": 1.0, "dosage_ml": 5, "timestamp": int(time.time())}
        client.publish(topic_cmd, json.dumps(cmd), qos=1)
        st.success("Manual command sent")

st.markdown("### Log (sample)")
st.text("Commands are published to MQTT broker.")
