# 🚗 License Plate Recognition & Route Distance Estimator

This Python GUI application uses OpenCV and EasyOCR to detect car license plates from images or webcam input. It cross-checks the license number with a CSV database for vehicle service history and calculates the shortest distance between cities based on user input.

---

## 🧠 Features

- 📸 **Image & Webcam Input**: Select an image or use live webcam to scan license plates.
- 🔍 **OCR Recognition**: Extracts license plate text using `easyocr`.
- 📋 **Car Info Lookup**: Matches detected number with a CSV database to show car age & servicing info.
- 🌐 **Route Distance Calculator**: Enter start and end cities to get the road distance.
- 🎨 **Tkinter GUI**: Clean, centered UI layout with intuitive interaction.

---

## 🛠️ Technologies Used

| Technology | Description |
|------------|-------------|
| Python     | Core programming language |
| Tkinter    | GUI framework |
| OpenCV     | Image processing and webcam handling |
| EasyOCR    | Optical Character Recognition |
| CSV        | Car data storage |
| PIL (Pillow) | Image display in GUI |

---

## 📦 Installation

1. **Clone the Repository**  
   ```bash
   git clone https://github.com/your-username/license-plate-recognition.git
   cd license-plate-recognition
