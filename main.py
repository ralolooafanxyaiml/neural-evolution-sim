# --- main.py ---
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical

# OWN MODULS
from database import ANIMAL_DATABASE, THREAT_DATABASE, EVOLUTION_MAPPING, ATTRIBUTE_CATEGORIES
from evolution_model import EvolutionModel

# --- 1. DATA LOADING ---
github_data_url = "https://raw.githubusercontent.com/ralolooafanxyaiml/neural-evolution-sim/refs/heads/main/data.csv"
df = pd.read_csv(github_data_url)

X = df[["METABOLISM", "SKIN", "HABITAT", "SIZE", "DIET", "THREAT"]]
y = df["EVOLUTION_TARGET"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
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
    X_train_scaled, y_train_encoded, 
    epochs=50, batch_size=32, 
    X_test=X_test_scaled, y_test=y_test_encoded
)

final_accuracy = evolution_sim.evaluate_model(X_test_scaled, y_test_encoded)

# --- 3. MACHINE STARTING  ---
def start_game_interface():
    print("\n\n####################################################")
    print("#      NEURAL EVOLUTION ENGINE (V1.0) - READY  #")
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
                features = ANIMAL_DATABASE[user_animal]
                animal_name_display = user_animal.capitalize()
                print(f"   Organism Selected: {animal_name_display}")
                break
            else:
                print(f"   Unknown Animal. Try: Lion, Wolf, Snake, Shark...")

        current_evolution_attributes = {} 

        # THREATS AND EVOLUTIONS
        while True:
            print(f"\n   --- Current Organism: {animal_name_display} ---")
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

            # PREDICT
            input_vector = np.array([features + [float(threat_id)]])
            input_scaled = scaler.transform(input_vector)
            
            predicted_id = evolution_sim.predict_id(input_scaled)

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

    start_game_interface()



