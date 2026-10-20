Vehicle Emission Monitoring System (VEMS)
🚦 Project Overview
This project, developed during an industrial attachment at the Centre for Robotics and Biomedical Engineering (Dedan Kimathi University of Technology), presents a low-cost, portable system for monitoring and identifying pollutant vehicles on the road.

By leveraging Computer Vision and Deep Learning, the system detects smoke emissions from vehicles in real-time. It utilizes a Raspberry Pi 3, a USB Camera, a PIR sensor for motion detection, and an NDIR sensor for CO2 monitoring.

📄 Abstract
Air pollution, particularly from vehicular emissions, poses significant health and environmental risks. Traditional monitoring methods (roadside inspections, fixed sensors) are expensive, time-consuming, and reactive. This project proposes an automated, proactive solution using a Binary Classification Convolutional Neural Network (CNN) trained with TensorFlow.

The system captures images of passing vehicles, classifies them as "pollutant" or "non-pollutant" based on visible smoke, and logs CO2 levels. Deployed on edge hardware, this solution offers a scalable and affordable tool for environmental monitoring and policy enforcement.

✨ Key Features
Real-time Detection: Identifies smoke-emitting vehicles instantly using a trained CNN model.

Low-Cost Hardware: Built on Raspberry Pi 3, making it accessible for developing regions.

Automated Data Collection: PIR sensor triggers image capture and CO2 measurement upon vehicle detection.

Edge Computing: Runs the machine learning model locally on the device without needing cloud infrastructure.

🛠️ System Architecture
Hardware Components
Raspberry Pi 3: Central processing unit.

Camera Module: Captures high-resolution images of vehicles.

PIR Sensor: Detects motion to trigger the system.

NDIR Sensor: Measures CO2 concentration in parts per million (ppm).

Software Stack
Language: Python

Framework: TensorFlow / Keras

Model: Convolutional Neural Network (CNN) - Sequential Architecture

Libraries: OpenCV, NumPy, Pandas

🧠 Methodology
1. Data Collection
Images were captured using a USB camera connected to the Raspberry Pi.

A dataset was created containing labeled images: smoky (pollutant) and non-smoky (non-pollutant).

CO2 levels were logged simultaneously via the NDIR sensor.

Data Augmentation: Techniques like rotation, flipping, and grayscaling were applied to increase dataset diversity and model robustness.

2. Model Architecture
The CNN model was built using TensorFlow's Sequential API:

Conv2D Layers: 4 layers with 64 filters (3x3 kernel) and ReLU activation.

MaxPooling2D: Reduced spatial dimensions.

Flatten: Converted 2D feature maps to 1D vectors.

Dense Layers: 512 units (ReLU) and Output layer (Sigmoid for binary classification).

Total Params: ~6.5 Million.

3. System Integration
The trained model was deployed to the Raspberry Pi. The workflow involves:

PIR sensor detects a vehicle.

Camera captures an image.

CNN classifies the image.

NDIR sensor records CO2 levels.

Data is logged to a CSV file with timestamps.

📊 Results & Challenges
Results
The system demonstrated high accuracy in classifying vehicles under various environmental conditions. The integration of low-cost hardware proved feasible for real-time edge computing applications.

Challenges Encountered
Environmental Variability: Lighting conditions (glare, shadows) and weather affected image quality.

Hardware Constraints: The Raspberry Pi 3 experienced processing delays, particularly with sensor synchronization.

Data Imbalance: The dataset had uneven classes (pollutant vs. non-pollutant), requiring careful handling during training.

🚀 Future Work & Recommendations
Hardware Upgrade: Transition to Raspberry Pi 4 or 5 for faster processing speeds.

Dataset Expansion: Include a wider variety of vehicles and environmental conditions to improve model generalization.

Model Optimization: Implement Transfer Learning (e.g., MobileNet) to reduce computational load and improve accuracy.

Sensor Fusion: Improve the synchronization between the PIR sensor and camera to reduce latency.

📚 References
Abadi, M., et al. (2016). TensorFlow: A system for large-scale machine learning.

Chollet, F. (2017). Deep Learning with Python.

Goodfellow, I., et al. (2016). Deep Learning. MIT Press.

Krizhevsky, A., et al. (2012). ImageNet classification with deep convolutional neural networks.

👥 Credits
Author: Gichanga Beth Mwara
Institution: Dedan Kimathi University of Technology
Department: Electrical and Electronics Engineering
Attachment Period: 8th January – 29th March 2024

Supervisors:

Dr. Joseph Muguro

Dr. Waweru Njeri

Project Partner: Faith Wanjiru Muchira
