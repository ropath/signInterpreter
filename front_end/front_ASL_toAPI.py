import streamlit as st
import cv2 as cv
import mediapipe as mp
import numpy as np
from PIL import Image
from io import BytesIO
import requests

def calc_bounding_rect(image, landmarks):
    padding = 60
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)
    return [x-padding, y-padding, x + w + padding, y + h + padding]

def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]), (0, 0, 0), 1)
    return image

def extract_hand(source_image):
    image = np.array(Image.open(source_image).convert("RGB"))
    debug_image = image.copy()
    image = cv.cvtColor(image, cv.COLOR_RGB2BGR)
    results = hands.process(image)

    if results.multi_hand_landmarks is not None:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            brect = calc_bounding_rect(debug_image, hand_landmarks)
            x_min, y_min, x_max, y_max = brect
            hand_region = image[y_min+1:y_max, x_min+1:x_max]
            debug_image = draw_bounding_rect(True, debug_image, brect)
            return debug_image, hand_region
    return None, None

def send_to_api(hand_region):
    is_success, buffer = cv.imencode(".jpg", hand_region)
    if is_success:
        img_bytes = BytesIO(buffer)

        response = requests.post("https://sign-interpreter-app-373962339093.europe-west1.run.app/predict",
            files={"file": ("hand_image.jpg", img_bytes, "image/jpeg")}
        )

        if response.status_code == 200:
            prediction = response.json()
            return prediction
        else:
            st.error("API Error: " + response.json().get("detail", "Unknown error"))

st.title("Show hands!")
st.write("Take a picture with the computer camera, or upload a file.")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)

camera_image = st.camera_input("Take a picture")
hand_region = None

if camera_image is not None:
    processed_image, hand_region = extract_hand(camera_image)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    processed_image, hand_region = extract_hand(uploaded_file)

if hand_region is not None:
    col1, col2 = st.columns(2)
    with col1:
        st.image(processed_image, caption="Original", use_column_width=True)
    with col2:
        st.image(hand_region, caption="Hand region")

    prediction = send_to_api(hand_region)
    if prediction:
        st.write(f"Prediction: {prediction['prediction']}")
        st.write(f"Confidence: {prediction['confidence']:.2f}")
else:
    st.write("No hand detected in the image.")
