# Neural Evolution Engine V2.0: Multi-Modal AI Simulation

### Project Overview
**Neural Evolution Engine V2.0** is an advanced Deep Learning application built from scratch using **TensorFlow** and **Keras**. It simulates evolutionary biology principles by predicting the optimal adaptation strategy for a species when faced with catastrophic environmental threats.

**What's New in V2.0?**
Unlike traditional simulations, this engine features a **Hybrid Multi-Input Neural Network** capable of processing both **Biological Data (Text/Numerical)** and **Environmental Imagery (Visual Data)** simultaneously. The AI "sees" the threat (e.g., an image of a blizzard) and "knows" the animal's biology to make a prediction.

---

### Technical Architecture (Dual-Core Brain)

The model utilizes a sophisticated **Multi-Input Functional API** architecture that merges two distinct neural networks:

#### 1. Visual Branch (The "Eye" - CNN)
* **Input:** 64x64 RGB Images of environmental threats.
* **Layers:** 2x Convolutional Layers (Conv2D) + MaxPooling2D for feature extraction.
* **Function:** Analyzes visual patterns (snow, fire, toxic waste) to identify the nature of the threat.

#### 2. Biological Branch (The "Brain" - ANN)
* **Input:** Encoded biological features (Metabolism, Skin Type, Habitat, Size, Diet).
* **Layers:** Dense Layers (32 -> 16 Neurons) with ReLU activation.
* **Function:** Processes the organism's physiological constraints.

#### 3. The Fusion Layer (Decision Making)
* **Concatenate:** Merges the visual features from the CNN and the biological data from the ANN.
* **Output Layer:** Softmax activation generating a probability distribution across 6 evolutionary outcomes.

---

### Tech Stack
* **Deep Learning:** TensorFlow, Keras (Functional API)
* **Computer Vision:** OpenCV (Image Preprocessing)
* **Data Engineering:** Pandas, NumPy
* **Preprocessing:** Scikit-Learn (StandardScaler, One-Hot Encoding)

---

### How It Works (Hybrid Modes)

The simulation offers two distinct modes of interaction:

1.  **Text Mode (Classic):** The user types the threat name (e.g., "Ice Age"), and the system maps it to a threat category.
2.  **Visual Mode (New!):** The user uploads an image file (or the system selects a random simulation image), and the **CNN** analyzes the visual data to determine the threat.

---

### Usage

To run the simulation locally, clone the repository and install the required dependencies:

1. **Clone the Repository**
```bash
git clone [https://github.com/ralolooafanxyaiml/Neural-Evolution-Engine.git](https://github.com/YOUR_USERNAME/Neural-Evolution-Engine.git)
cd Neural-Evolution-Engine
pip install tensorflow pandas numpy scikit-learn opencv-python
python setup.py
python main.py
```

Data Sources & Acknowledgements
This project uses the following public datasets for training the Visual Threat Detection (CNN) module:

Intel Image Classification by Puneet Bansal (Cold/Ice)

Natural Disaster Images by Aseem Arora (Heat/Fire)

Garbage Classification by Sashaank Sekar (Toxin/Pollution)

Underwater Image Classification by Great Sharma (Airless/Aquatic)

US Drought Data (Scarcity)

Developed by Mustafa İlker Aktaş - Global AI Contributor
