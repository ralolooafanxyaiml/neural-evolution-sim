# --- evolution_model.py ---
# Imported Model, Input Dense, Conv2D, Maxpooling2D, Flatten, concatenate, Adam.
import numpy as np
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, concatenate
from tensorflow.keras.optimizers import Adam

class EvolutionModel:
    def __init__(self, input_dim, output_dim, mapping):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.evolution_map = mapping
        self.model = self.build_model()

    # Added CNN, adjusted AAN for combine it with CNN.
    def build_model(self):
        Input_Visual = Input(shape=(64, 64, 3), name="Visual_Input")

        CNN = Conv2D(32, (3, 3), activation="relu")(Input_Visual)
        CNN = MaxPooling2D(pool_size=(2, 2))(CNN)
        CNN = Conv2D(64, (3, 3), activation="relu")(CNN)
        CNN = MaxPooling2D(pool_size=(2, 2))(CNN)
        CNN = Flatten()(CNN)

        Input_Biological = Input(shape=(self.input_dim,), name="Biological_Input")
        
        ANN = Dense(32, activation="relu")(Input_Biological)
        ANN = Dense(16, activation="relu")(ANN)

        combined = concatenate([CNN, ANN])

        CNNandANN = Dense(32, activation="relu")(combined)
        output = Dense(self.output_dim, activation="softmax", name="Evolution_Output")(CNNandANN)

        model = Model(inputs=[Input_Visual, Input_Biological], outputs=output)
        
        return model

    # Adam is more sensitive now for CNN.
    def compile_model(self):
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss="categorical_crossentropy", metrics=["accuracy"])

    def fit_model(self, X_train, y_train, epochs, batch_size, X_test, y_test):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), verbose=1)

    def predict_id(self, input_features):
        prediction_probabilities = self.model.predict(input_features, verbose=0)
        predicted_id = np.argmax(prediction_probabilities, axis=1)[0]
        return predicted_id

    def evaluate_model(self, X_test, y_test):
        loss, accuracy = self.model.evaluate(X_test, y_test, verbose=0)

        return accuracy
