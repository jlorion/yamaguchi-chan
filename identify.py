import cv2
import numpy as np
# Load your ONNX model
net = cv2.dnn.readNetFromONNX("model.onnx")


# Define your labels (modify as needed)
class_names = ['plastic', 'metal', 'glass', 'paper']  # Example classes

# Open the webcam
cap = cv2.VideoCapture(0)  # 0 = default webcam

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame for ONNX model
    # input_blob = cv2.resize(frame, (224, 224))                 # Resize to match model input
    # input_blob = input_blob.astype(np.float32) / 255.0         # Normalize to 0-1
    # input_blob = np.transpose(input_blob, (2, 0, 1))           # HWC → CHW
    # input_blob = np.expand_dims(input_blob, axis=0)            # Add batch dimension
    # input_blob = np.ascontiguousarray(input_blob)
    # version 1
    # blob = cv2.dnn.blobFromImage(
    #     frame, 
    #     scalefactor=1.0/255.0,
    #     size=(224, 224),  # Match model's input
    #     mean=(0.5, 0.5, 0.5),
    #     swapRB=True,      # Convert BGR to RGB
    #     crop=False
    # )
    # version2
    blob = cv2.dnn.blobFromImage(
        frame,
        scalefactor=1.0 / 255.0,
        size=(224, 224),
        mean=(0.485, 0.456, 0.406),
        swapRB=True,
        crop=False
    )

    # Apply standard deviation scaling (optional)
    std = (0.229, 0.224, 0.225)  # ImageNet standard deviation values
    # blob[0] /= np.array(std).reshape(3, 1, 1)
    for i in range(3):  # For each channel
        blob[0, i] = blob[0, i] / std[i]


    # Set input and run inference
    net.setInput(blob)
    output = net.forward()

    
    class_id = int(np.argmax(output))
    confidence = float(np.max(output))
    print(output)

    # 🏷️ Format label
    label = f"{class_id} ({confidence*100:.1f}%)"

    # Display label on frame
    cv2.putText(frame, f'Predicted: {label}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Live Classification', frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()