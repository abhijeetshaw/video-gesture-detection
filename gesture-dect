import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# Function to preprocess frame
def preprocess_frame(frame):
    resized_frame = cv2.resize(frame, (300, 300))
    rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
    normalized_frame = rgb_frame / 255.0
    input_frame = np.expand_dims(normalized_frame, axis=0)
    return input_frame

# Function to detect gesture
def detect_gesture(frame, model):
    input_frame = preprocess_frame(frame)
    predictions = model.predict(input_frame)
    # Assuming gesture class index is 0
    if np.argmax(predictions) == 0:
        return True
    return False

# Load pre-trained SSD MobileNet V2 model
base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(300, 300, 3))
base_model.trainable = False

# Add custom detection head
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
predictions = layers.Dense(1, activation='sigmoid')(x)  # Binary classification (gesture or not)
model = models.Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=optimizers.Adam(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Assuming you have labeled data and data generators defined
train_generator = ...
val_generator = ...
test_generator = ...

# Train the model
model.fit(train_generator,
          epochs=num_epochs,
          validation_data=val_generator)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_generator)
print("Test Accuracy:", test_acc)

# Load test video
cap = cv2.VideoCapture("path/to/test_video.mp4")

# Define font and text properties for annotation
font = cv2.FONT_HERSHEY_SIMPLEX
org = (50, 50)
font_scale = 1
color = (0, 255, 0)  # Bright green color
thickness = 2

# Process video frames
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Perform gesture detection
    gesture_detected = detect_gesture(frame, model)
    
    # Annotate frame if gesture is detected
    if gesture_detected:
        annotated_frame = cv2.putText(frame, 'DETECTED', org, font, font_scale, color, thickness, cv2.LINE_AA)
    else:
        annotated_frame = frame
    
    # Display annotated frame
    cv2.imshow('Gesture Detection', annotated_frame)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
