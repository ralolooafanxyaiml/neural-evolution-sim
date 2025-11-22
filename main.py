# --- main.py ---
# imported os, cv2.
import pandas as pd
import numpy as np
import random
import os
import cv2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

# OWN MODULS
from database import ANIMAL_DATABASE, THREAT_DATABASE, EVOLUTION_MAPPING, ATTRIBUTE_CATEGORIES
from evolution_model import EvolutionModel

# --- 1. VISUAL DATA LOAD --- added new

def load_images_and_threats(df):
    image_data = []
    valid_indices = []

    base_dir = "./visual_datasets"
    folder_map = {1: "1_COLD", 2: "2_HEAT", 3: "3_TOXIN", 4: "4_SCARCITY", 5: "5_AIRLESS"}
    threat_images = {k: [] for k in folder_map}

    for t_id, folder in folder_map.items():
        path = os.path.join(base_dir, folder)
        if os.path.exists(path):
            files = os.listdir(path)[:50]
            for f in files:
                try:
                    img = cv2.imread(os.path.join(path, f))
                    img = cv2.resize(img, (64, 64))
                    threat_images[t_id].append(img / 255.0)
                except: pass

    for index, row in df.iterrows():
        t = row['THREAT']
        if t in threat_images and len(threat_images[t]) > 0:
            image_data.append(random.choice(threat_images[t]))
            valid_indices.append(index)
    return np.array(image_data), df.iloc[valid_indices]
    
# --- 1. DATA ENCODING --- # df is changed as raw_df because of df is used at visual data load. # added image train/test. # epochs deceased for CNN.
github_data_url = "https://raw.githubusercontent.com/ralolooafanxyaiml/neural-evolution-sim/refs/heads/main/data.csv"
raw_df = pd.read_csv(github_data_url)

X_images, df = load_images_and_threats(raw_df)

X = df[["METABOLISM", "SKIN", "HABITAT", "SIZE", "DIET", "THREAT"]]
y = df["EVOLUTION_TARGET"]

X_img_train, X_img_test, X_test, X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_train_encoded = to_categorical(y_train, num_classes=6)
y_test_encoded = to_categorical(y_test, num_classes=6)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- 2. AI MODEL ---
input_feature_count = X_train_scaled.shape[1]
output_class_count = y_train_encoded.shape[1]

evolution_sim = EvolutionModel(
    input_dim=input_feature_count, 
    output_dim=output_class_count, 
    mapping=EVOLUTION_MAPPING
)

evolution_sim.compile_model()
evolution_sim.fit_model(
    [X_train_scaled, X_img_train] y_train_encoded, 
    epochs=20, batch_size=32, 
    X_test=[X_test_scaled, X_img_test], y_test=y_test_encoded
)

final_accuracy = evolution_sim.evaluate_model(X_test_scaled, y_test_encoded)

# --- 3. MACHINE STARTING  --- # changed features too work CNN properly. # Visual Threat Input Mode Added!!
def start_engine_interface():
    print("\n\n####################################################")
    print("#      NEURAL EVOLUTION ENGINE (V2.0) - READY  #") #V1.0 -) V2.0
    print("####################################################")
    
    while True:
        print("\n--- NEW EVOLUTIONARY LINEAGE STARTED ---")
        features = []
        animal_name_display = ""
        
        # ANIMAL CHANGE
        while True:
            user_animal = input("\n>> ENTER ANIMAL NAME (or 'exit' to close): ").lower().strip()
            if user_animal == 'exit':
                print("Goodbye!")
                return 

            if user_animal in ANIMAL_DATABASE:
                features = ANIMAL_DATABASE[user_animal][:5]
                animal_name_display = user_animal.capitalize()
                print(f"   Organism Selected: {animal_name_display}")
                break
            else:
                print(f"   Unknown Animal. Try: Lion, Wolf, Snake, Shark...")

        current_evolution_attributes = {} 

            print("\nSELECT THREAT INPUT MODE:")
            print("   [1] TEXT INPUT")
            print("   [2] VISUAL INPUT")
            mode = input(">> Mode (1/2): ").strip()

        # THREATS AND EVOLUTIONS. # Added Visual Mode and Selecting Mode.
        while True:
            print(f"\n   --- Current Organism: {animal_name_display} ---")

            input_img = None
            threat_desc = "Unknown"

            if mode == "1":
                user_threat = input(">> ENTER THREAT (or type 'quit' to change animal): ").lower().strip()
                if user_threat == 'quit':
                    break 
            
                threat_id = None
                for key in THREAT_DATABASE:
                    if key in user_threat:
                       threat_id = THREAT_DATABASE[key]
                       break
            
                if not threat_id:
                print("   Unknown Threat. Try: Cold, Heat, Virus, Predator...")
                continue

                input_img = np.zeros((1, 64, 64, 3))
                threat_desc = user_threat.upper()

            elif mode == '2':
                user_path = input(">> ENTER IMAGE PATH (or 'quit'): ").strip()
                if user_path == "quit": break

                try:
                    img = cv2.imread(user_path)
                    img = cv2.resize(img, (64, 64))
                    input_img = np.array([img / 255.0])
                    threat_desc = "VISUAL THREAT"
                except:
                    print("   Image Error! Using blank.")
                    input_img = np.zeros((1, 64, 64, 3))

            # PREDICT # Added visual predicts!!
            biological_input = np.array([features]) # Threat ID yok, sadece biyolojik
            input_scaled = scaler.transform(biological_input)

            probs = evolution_sim.model.predict([input_img, input_scaled], verbose=0)
            predicted_id = np.argmax(probs, axis=1)[0]

            # RESULTS
            evolution_options = EVOLUTION_MAPPING.get(predicted_id, ["Error"])
            final_description = random.choice(evolution_options)
            
            category = ATTRIBUTE_CATEGORIES.get(predicted_id, "UNKNOWN")
            current_evolution_attributes[category] = final_description
            
            print(f"\n   EVOLUTION TRIGGERED: {category}")
            print(f"   Result: {final_description}")
            
            print("\n   [CURRENT DNA INVENTORY]:")
            if not current_evolution_attributes:
                print("      (No mutations yet)")
            else:
                for cat, desc in current_evolution_attributes.items():
                    print(f"      * {cat}: {desc}")
            print("-" * 50)

# START
if __name__ == "__main__":

    start_engine_interface()





