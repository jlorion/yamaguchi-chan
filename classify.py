import cv2
import numpy as np
import time

# Load your ONNX model
net = cv2.dnn.readNetFromONNX("model.onnx")

# Create shorter display names for UI
display_names = [
    'Others',
    'Plastic'
]

# Open the webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Add smoothing for predictions
last_predictions = []
smoothing_window = 5

while True:
    ret, frame = cap.read()
    if not ret:
        break
       
    # Create a copy for display
    display_frame = frame.copy()
    
    # Process center region of frame for better focus
    h, w = frame.shape[:2]
    center_size = min(h, w) * 0.8  # Use 80% of frame
    x_offset = int((w - center_size) / 2)
    y_offset = int((h - center_size) / 2)
    
    # Draw ROI rectangle
    roi_start = (x_offset, y_offset)
    roi_end = (w - x_offset, h - y_offset)
    cv2.rectangle(display_frame, roi_start, roi_end, (0, 255, 0), 2)
    
    # Crop to region of interest
    roi = frame[y_offset:h-y_offset, x_offset:w-x_offset]
    
    # Preprocess using the corrected approach
    blob = cv2.dnn.blobFromImage(
        roi,
        scalefactor=1.0 / 255.0,
        size=(224, 224),
        mean=(0.485, 0.456, 0.406),
        swapRB=True,
        crop=False
    )
    
    # Apply standard deviation scaling (required for ImageNet pretrained models)
    # std = (0.229, 0.224, 0.225)
    # for i in range(3):
    #     blob[0, i] = blob[0, i] / std[i]
    
    # Set input and run inference
    net.setInput(blob)
    output = net.forward()
    
    # Get class predictions
    class_id = int(np.argmax(output[0]))
    
    confidence = float(output[0][class_id])
    print(class_id)
    
    # The model output might be indexed differently than our class_names
    # Note: Based on your list, it appears index 0 is tekstil (Textile), etc.
    # Adjust if your model uses different indexing
    
    # Apply smoothing
    last_predictions.append(class_id)
    if len(last_predictions) > smoothing_window:
        last_predictions.pop(0)
    
    # Use most common prediction in the window
    predicted_class = max(set(last_predictions), key=last_predictions.count)
    
    # Get confidence for the smoothed prediction
    smoothed_confidence = float(output[0][predicted_class])
    
    # Only show confident predictions
    prediction_text = "Analyzing..."
    if smoothed_confidence > 0.5:  # Confidence threshold
        prediction_text = f"{display_names[predicted_class]} ({smoothed_confidence*100:.1f}%)"
    
    # Add UI elements
    # Title bar
    cv2.rectangle(display_frame, (0, 0), (w, 60), (50, 50, 50), -1)
    cv2.putText(display_frame, "Waste Classification", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Prediction box
    cv2.rectangle(display_frame, (0, h-60), (w, h), (50, 50, 50), -1)
    cv2.putText(display_frame, f"Prediction: {prediction_text}", (20, h-20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Instructions
    cv2.putText(display_frame, "Press 'q' to quit", (w-200, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Show the frame
    cv2.imshow('Waste Classification', display_frame)
    
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()