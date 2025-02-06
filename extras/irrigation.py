import time
from ultralytics import YOLO
import cv2
import pandas as pd

# Load YOLO model for pest detection
model = YOLO('yolov8n.pt')

# Simulate loading soil health dataset
soil_health_data = pd.DataFrame({
    'humidity': [35, 40, 25, 30],
    'temperature': [25, 27, 26, 28],
    'rainfall': [4, 3, 2, 1]
})

# Function to detect pests
def detect_pests(image_path):
    image = cv2.imread(image_path)
    results = model.predict(source=image)
    print(results)
    # Loop through detections and check for pest classes
    pests_detected = False
    for r in results:
        print('hi1')
        for box in r.boxes:
            print(box.cls)
            cls = int(box.cls[0])
            print(cls)  # Class of the detected object
            if cls in [75,14]:

                print('hi3')  # Example: IDs 0, 1, 2 could correspond to pests
                pests_detected = True
                break
            print(cls)
        if pests_detected:
            break

    return pests_detected

# Function to analyze soil health
def analyze_soil_health(soil_path):
    avg_humidity = soil_path['Humidity'].mean()
    avg_temperature = soil_path['Temperature'].mean()
    avg_rainfall = soil_path['Rainfall'].mean()
    
    if avg_humidity < 30 and avg_rainfall < 5:  # Example thresholds
        return True  # Need irrigation non fertile
    else:
        return False  # No irrigation needed fertile

# Simulated function to control irrigation system
def water_plants(duration=10):
    print("Simulating: Watering the plants...")
    time.sleep(duration)  # Simulate watering duration
    print("Simulating: Stopped watering.")

# Main program loop (for Colab)
try:
    # Path to the pest detection image (you can upload an image in Colab and provide the path)
    image_path = "healthy_leaf1.jpg"
    soil_path=pd.read_csv("soil_final_data.csv")

    # Pest detection check
    if detect_pests(image_path):
        print("Pests detected. Skipping irrigation.")
    else:
        # Soil health assessment
        if analyze_soil_health(soil_path):
            print("Soil health requires irrigation.")
            water_plants()  # Simulate irrigation
        else:
            print("No irrigation needed based on soil health.")

except Exception as e:
    print(f"Error: {e}")