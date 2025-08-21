import sys
import cv2
import mediapipe as mp
import numpy as np
import pickle
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap, QColor, QPainter, QFont, QPainterPath, QPen, QIcon
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout, QHBoxLayout, 
                           QGraphicsScene, QGraphicsView, QGraphicsPixmapItem, QFrame,
                           QPushButton, QStyle)

# Load the trained model
model_dict = pickle.load(open('C:\\Mahmoud_Saeed\\My_projects\\pattern project\\sign\\model.p', 'rb'))
model = model_dict['model']

# Define labels dictionary for ASL letters
labels_dict = {
    0: "A",
    1: "B",
    2: "G",
    3: "L",
    4: "S",
    5: "Space",
    6: "nothing",
    7: "Z",
    8: "Y",
    9: "W",
    10 : "O",
    11 : "P",
    12 : "N",
    13 : "J",
   }

# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Initialize the hands object
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def draw_colored_landmarks(frame, hand_landmarks, theme_color):
    # Define colors for each finger (in BGR format)
    finger_colors = {
        'THUMB': (255, 0, 0),      # Blue
        'INDEX': (0, 255, 0),      # Green
        'MIDDLE': (0, 0, 255),     # Red
        'RING': (255, 0, 255),     # Magenta
        'PINKY': (0, 255, 255)     # Yellow
    }
    
    # Finger landmark indices
    finger_landmarks = {
        'THUMB': [1, 2, 3, 4],
        'INDEX': [5, 6, 7, 8],
        'MIDDLE': [9, 10, 11, 12],
        'RING': [13, 14, 15, 16],
        'PINKY': [17, 18, 19, 20]
    }
    
    # Draw palm landmarks in white
    palm_landmarks = [0]  # Wrist
    h, w = frame.shape[:2]
    
    # Draw palm landmark
    for idx in palm_landmarks:
        pos = hand_landmarks.landmark[idx]
        cx, cy = int(pos.x * w), int(pos.y * h)
        cv2.circle(frame, (cx, cy), 5, theme_color, -1)
    
    # Draw fingers with different colors
    for finger, indices in finger_landmarks.items():
        color = finger_colors[finger]
        
        # Draw landmarks for each finger
        for idx in indices:
            pos = hand_landmarks.landmark[idx]
            cx, cy = int(pos.x * w), int(pos.y * h)
            cv2.circle(frame, (cx, cy), 5, color, -1)
            
            # Draw connections between landmarks
            if idx > 0:
                prev_pos = hand_landmarks.landmark[idx - 1]
                prev_cx, prev_cy = int(prev_pos.x * w), int(prev_pos.y * h)
                cv2.line(frame, (prev_cx, prev_cy), (cx, cy), color, 2)

class ThemeColors:
    def __init__(self, is_dark=True):
        self.update_colors(is_dark)
    
    def update_colors(self, is_dark):
        if is_dark:
            # Dark theme with blue and purple tones
            self.bg_primary = "#1a1a2e"        # Deep blue-purple
            self.bg_secondary = "#232340"      # Lighter blue-purple
            self.border = "#4040bf"            # Medium blue
            self.text_primary = "#e0e0ff"      # Light blue-tinted white
            self.text_secondary = "#9090dd"    # Muted purple
            self.accent = "#6b46c1"            # Rich purple
            self.landmark_color = (200, 180, 255)  # Light purple in BGR
            self.connection_color = (180, 160, 220) # Muted purple in BGR
        else:
            # Light theme with blue and purple tones
            self.bg_primary = "#f0f0ff"        # Very light blue
            self.bg_secondary = "#ffffff"      # White
            self.border = "#8a7eee"            # Soft purple
            self.text_primary = "#2c1f56"      # Dark purple
            self.text_secondary = "#5448a8"    # Medium purple
            self.accent = "#6b46c1"            # Rich purple
            self.landmark_color = (130, 80, 180)   # Purple in BGR
            self.connection_color = (150, 100, 200) # Light purple in BGR

class CircularLabel(QLabel):
    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setMinimumSize(150, 150)
        self.setMaximumSize(150, 150)
        self.setAlignment(Qt.AlignCenter)
        self.theme_colors = ThemeColors()

    def update_theme(self, is_dark):
        self.theme_colors.update_colors(is_dark)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw circle
        painter.setPen(QPen(QColor(self.theme_colors.border), 3))
        painter.setBrush(QColor(self.theme_colors.bg_secondary))
        painter.drawEllipse(10, 10, 130, 130)

        # Draw text
        painter.setFont(QFont("Arial", 48, QFont.Bold))
        painter.setPen(QColor(self.theme_colors.text_primary))
        painter.drawText(self.rect(), Qt.AlignCenter, self.text())

class ASLRecognitionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.is_dark_mode = True
        self.theme_colors = ThemeColors(self.is_dark_mode)
        self.initUI()

    def initUI(self):
        self.setWindowTitle("ABLG Letter Recognition")
        self.setGeometry(100, 100, 1200, 800)
        
        # Main layout
        self.main_layout = QVBoxLayout()
        self.main_layout.setSpacing(20)
        self.main_layout.setContentsMargins(30, 30, 30, 30)

        # Header layout for title and theme switch
        header_layout = QHBoxLayout()

        # Title
        self.title = QLabel("ABLG Letter Recognition")
        
        # Theme switch button
        self.theme_button = QPushButton()
        self.theme_button.setFixedSize(40, 40)
        self.theme_button.clicked.connect(self.toggle_theme)
        header_layout.addWidget(self.title, stretch=1)
        header_layout.addWidget(self.theme_button)

        self.main_layout.addLayout(header_layout)

        # Content layout
        self.content_layout = QHBoxLayout()
        self.content_layout.setSpacing(30)

        # Camera panel
        self.camera_panel = QFrame()
        camera_layout = QVBoxLayout(self.camera_panel)

        # Camera view
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setRenderHint(QPainter.Antialiasing)
        self.view.setRenderHint(QPainter.SmoothPixmapTransform)
        self.view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.view.setMinimumSize(800, 600)
        camera_layout.addWidget(self.view)

        self.content_layout.addWidget(self.camera_panel, stretch=2)

        # Right panel
        self.right_panel = QFrame()
        right_layout = QVBoxLayout(self.right_panel)

        # Status label
        self.status_label = QLabel("Current Status")
        right_layout.addWidget(self.status_label)

        # Circular letter display
        self.letter_display = CircularLabel("?")
        right_layout.addWidget(self.letter_display, alignment=Qt.AlignCenter)
        
        # Detected letter label
        self.letter_label = QLabel("Detected Letter")
        right_layout.addWidget(self.letter_label)

        right_layout.addStretch()
        self.content_layout.addWidget(self.right_panel, stretch=1)

        # Add content layout to main layout
        self.main_layout.addLayout(self.content_layout)
        self.setLayout(self.main_layout)

        # Initialize webcam
        self.cap = cv2.VideoCapture(0)

        # Start timer for frame updates
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(20)

        # Apply initial theme
        self.update_theme()

    def update_theme(self):
        # Update colors
        self.theme_colors.update_colors(self.is_dark_mode)
        
        # Update main window
        self.setStyleSheet(f"""
            QWidget {{
                background-color: {self.theme_colors.bg_primary};
            }}
        """)

        # Update title
        self.title.setStyleSheet(f"""
            font-size: 32px;
            font-weight: bold;
            color: {self.theme_colors.text_primary};
            padding: 15px;
            border-bottom: 2px solid {self.theme_colors.accent};
            margin-bottom: 10px;
        """)

        # Update theme button
        self.theme_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {self.theme_colors.bg_secondary};
                border: 2px solid {self.theme_colors.border};
                border-radius: 20px;
                padding: 5px;
            }}
            QPushButton:hover {{
                background-color: {self.theme_colors.accent};
            }}
        """)
        # Update button icon based on theme
        icon_name = "🌙" if self.is_dark_mode else "☀"
        self.theme_button.setText(icon_name)

        # Update panels
        panel_style = f"""
            QFrame {{
                background-color: {self.theme_colors.bg_secondary};
                border: 2px solid {self.theme_colors.border};
                border-radius: 15px;
                padding: 20px;
            }}
        """
        self.camera_panel.setStyleSheet(panel_style)
        self.right_panel.setStyleSheet(panel_style)

        # Update labels
        label_style = f"""
            color: {self.theme_colors.text_primary};
            font-size: 24px;
            margin-top: 10px;
        """
        self.status_label.setStyleSheet(label_style)
        self.letter_label.setStyleSheet(label_style)

        # Update letter display
        self.letter_display.update_theme(self.is_dark_mode)

    def toggle_theme(self):
        self.is_dark_mode = not self.is_dark_mode
        self.update_theme()

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        # Basic frame processing
        frame = cv2.flip(frame, 1)
        
        # Process frame for hand detection
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            data_aux = []
            x_ = []
            y_ = []

            for hand_landmarks in results.multi_hand_landmarks:
                # Use custom colored landmarks drawing
                draw_colored_landmarks(frame, hand_landmarks, self.theme_colors.landmark_color)
                
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    x_.append(x)
                    y_.append(y)

                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x
                    y = hand_landmarks.landmark[i].y
                    data_aux.append(x - min(x_))
                    data_aux.append(y - min(y_))

            try:
                data_aux = np.asarray(data_aux).reshape(1, -1)
                prediction = model.predict(data_aux)
                predicted_character = labels_dict.get(int(prediction[0]), '?')
                
                # Update letter display
                self.letter_display.setText(predicted_character)
                
                # Draw bounding box with theme-appropriate colors
                H, W = frame.shape[:2]
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) + 10
                y2 = int(max(y_) * H) + 10
                
                box_color = self.theme_colors.landmark_color
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
                
                # Corner accents
                corner_length = 20
                cv2.line(frame, (x1, y1), (x1 + corner_length, y1), box_color, 3)
                cv2.line(frame, (x1, y1), (x1, y1 + corner_length), box_color, 3)
                cv2.line(frame, (x2, y2), (x2 - corner_length, y2), box_color, 3)
                cv2.line(frame, (x2, y2), (x2, y2 - corner_length), box_color, 3)
                
            except Exception as e:
                print(f"Error in prediction: {e}")
                self.letter_display.setText("?")
        else:
            self.letter_display.setText("?")

        # Convert to RGB for Qt
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert frame to QImage and update display
        height, width, channel = frame_rgb.shape
        bytes_per_line = channel * width
        qt_image = QImage(frame_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)

        # Update canvas
        self.scene.clear()
        self.scene.addPixmap(pixmap)
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ASLRecognitionApp()
    window.show()
    sys.exit(app.exec_())


    