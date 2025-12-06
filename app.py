import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration

# --- 1. Invisible Cloak Processor ---
class CloakProcessor(VideoProcessorBase):
    def __init__(self):
        self.background = None
        self.capture_bg = False
        self.color_choice = 'Red'  # Default

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Capture background if requested
        if self.capture_bg:
            self.background = np.flip(img, axis=1)
            self.capture_bg = False
        
        # If no background captured yet, just return flipped image
        if self.background is None:
            return av.VideoFrame.from_ndarray(np.flip(img, axis=1), format="bgr24")

        # Processing logic
        img = np.flip(img, axis=1)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Define color ranges based on selection
        if self.color_choice == 'Red':
            lower1 = np.array([0, 120, 70])
            upper1 = np.array([10, 255, 255])
            lower2 = np.array([170, 120, 70])
            upper2 = np.array([180, 255, 255])
            mask1 = cv2.inRange(hsv, lower1, upper1)
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask = mask1 + mask2
        elif self.color_choice == 'Blue':
            mask = cv2.inRange(hsv, np.array([100, 150, 0]), np.array([140, 255, 255]))
        elif self.color_choice == 'Green':
            mask = cv2.inRange(hsv, np.array([35, 50, 50]), np.array([85, 255, 255]))
        elif self.color_choice == 'White':
            mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 55, 255]))
        else:
            mask = np.zeros(img.shape[:2], dtype=np.uint8)

        # Morphological operations
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))

        # Create final output
        mask_inv = cv2.bitwise_not(mask)
        res1 = cv2.bitwise_and(self.background, self.background, mask=mask)
        res2 = cv2.bitwise_and(img, img, mask=mask_inv)
        final_output = cv2.addWeighted(res1, 1, res2, 1, 0)

        return av.VideoFrame.from_ndarray(final_output, format="bgr24")

# --- 2. Gesture Game Processor ---
class GestureProcessor(VideoProcessorBase):
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
        self.mp_drawing = mp.solutions.drawing_utils
        self.start_x = None
        self.start_y = None
        self.command_text = "Waiting..."

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)
        
        height, width, _ = img.shape
        threshold = 50

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(img, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                
                # Index Finger Tip (Landmark 8)
                index_tip = hand_landmarks.landmark[8]
                cx, cy = int(index_tip.x * width), int(index_tip.y * height)
                
                cv2.circle(img, (cx, cy), 10, (0, 255, 0), -1)

                if self.start_x is None:
                    self.start_x, self.start_y = cx, cy
                
                diff_x = cx - self.start_x
                diff_y = cy - self.start_y

                if abs(diff_x) > threshold or abs(diff_y) > threshold:
                    if abs(diff_x) > abs(diff_y):
                        self.command_text = "RIGHT >>" if diff_x > 0 else "<< LEFT"
                    else:
                        self.command_text = "DOWN v" if diff_y > 0 else "^ UP"
                    
                    # Reset start point after detection to detect next swipe
                    self.start_x, self.start_y = cx, cy
        else:
            self.start_x = None # Reset if hand lost

        # Display command on screen
        cv2.putText(img, self.command_text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                    1, (0, 0, 255), 3)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Main App ---
def main():
    st.set_page_config(page_title="Web Magic", layout="wide")
    st.title("Web Magic 🧙‍♂️")
    
    # Permission instructions
    st.info("⚠️ When the camera starts, your browser will ask for permission. Click 'Allow'.")

    mode = st.sidebar.selectbox("Choose Mode", ["Invisible Cloak", "Gesture Test"])

    # Google STUN server for public access
    rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

    if mode == "Invisible Cloak":
        st.header("Invisible Cloak")
        color = st.sidebar.selectbox("Cloak Color", ["Red", "Blue", "Green", "White"])
        
        # Webrtc streamer
        ctx = webrtc_streamer(
            key="cloak", 
            video_processor_factory=CloakProcessor,
            rtc_configuration=rtc_config
        )

        # Pass color selection to processor
        if ctx.video_processor:
            ctx.video_processor.color_choice = color
        
        # Button to trigger background capture
        if st.button("📸 Capture Background (Step out of frame first!)"):
            if ctx.video_processor:
                ctx.video_processor.capture_bg = True
                st.success("Background capturing... Stay out for 1 second!")

    elif mode == "Gesture Test":
        st.header("Gesture Test")
        st.markdown("Swipe your index finger to see directions detected.")
        
        webrtc_streamer(
            key="gesture", 
            video_processor_factory=GestureProcessor,
            rtc_configuration=rtc_config
        )

if __name__ == '__main__':
    main()