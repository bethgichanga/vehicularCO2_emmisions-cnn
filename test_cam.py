import os
import cv2
import numpy as np
import tensorflow as tf

# Load the Keras model
model_path = '/home/pizero/tflite1/CNN_model2_jupyter.keras'  # Replace with the actual path
model = tf.keras.models.load_model(model_path)

# Initialize the camera
cap = cv2.VideoCapture(0)  # You may need to change the index if there are multiple cameras

# Preprocessing function
def preprocess_frame(frame):
    frame_resized = cv2.resize(frame, (512, 512))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    crop_size = min(frame_rgb.shape[0], frame_rgb.shape[1])
    frame_cropped = frame_rgb[(frame_rgb.shape[0] - crop_size) // 2 : (frame_rgb.shape[0] + crop_size) // 2,
                              (frame_rgb.shape[1] - crop_size) // 2 : (frame_rgb.shape[1] + crop_size) // 2]
    frame_normalized = frame_cropped / 255.0
    return frame_normalized

# Postprocessing function
def postprocess_output(output_data, original_frame):
    _, segmented_image = cv2.threshold(output_data[0], 0.5, 1, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    segmented_image = cv2.morphologyEx(segmented_image, cv2.MORPH_OPEN, kernel)
    segmented_image = cv2.morphologyEx(segmented_image, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(segmented_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = original_frame.copy()
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    result_image = cv2.addWeighted(original_frame, 0.7, overlay, 0.3, 0)
    return result_image

# Capture a frame from the camera in a loop
while True:
    ret, frame = cap.read()
    
    # Preprocess the frame
    input_data = preprocess_frame(frame)

    # Perform inference
    input_data = np.expand_dims(input_data, axis=0).astype(np.float32)
    prediction = model.predict(input_data)

    # Postprocess the output
    result_image = postprocess_output(prediction, frame)

    # Interpret the model output (example for binary classification)
    class_label = 'Smoke' if prediction[0][0] > 0.5 else 'Non-Smoke'

    # Display the results
    cv2.putText(result_image, f'Prediction: {class_label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Segmented Image', result_image)
    
    # Exit the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
