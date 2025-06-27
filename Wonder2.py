import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np
import imutils
import easyocr
import csv
import os

# Create Cars.csv if not exists
if not os.path.exists("Cars.csv"):
    Cars = [
        {"Car_Number": "IT20 BOM", "Age_Of_Car": 5, "Months_without_Servicing": 3},
        {"Car_Number": "MH12 AB1234", "Age_Of_Car": 11, "Months_without_Servicing": 7},
        {"Car_Number": "DL10 XY9876", "Age_Of_Car": 3, "Months_without_Servicing": 2},
    ]
    with open("Cars.csv", mode="w", newline='') as csvfile:
        fieldnames = ["Car_Number", "Age_Of_Car", "Months_without_Servicing"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in Cars:
            writer.writerow(row)


class LicensePlateRecognitionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("License Plate Recognition App")
        self.root.geometry("800x600")

        self.cars = self.load_car_details()
        self.start_point = ""
        self.end_point = ""

        self.routes = {
            ("Chennai", "Bangalore"): 350,
            ("Chennai", "Hyderabad"): 630,
            ("Bangalore", "Hyderabad"): 570,
            ("Bangalore", "Mumbai"): 980,
            ("Hyderabad", "Mumbai"): 710,
            ("Chennai", "Mumbai"): 1330,
            ("Delhi", "Mumbai"): 1410,
            ("Delhi", "Hyderabad"): 1560,
            ("Delhi", "Bangalore"): 1750,
            ("Delhi", "Chennai"): 2200,
        }
        self.routes.update({(to, frm): dist for (frm, to), dist in self.routes.items()})

        self.shortest_route_result = ""

        self.create_widgets()

    def load_car_details(self):
        cars = []
        try:
            with open("Cars.csv", mode='r') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    cars.append(row)
        except Exception as e:
            messagebox.showerror("Error", str(e))
        return cars

    def create_widgets(self):
        self.main_frame = tk.Frame(self.root, bg="#F5F5F5")
        self.main_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        self.processed_image_label = tk.Label(self.main_frame, bg="white")
        self.processed_image_label.pack(pady=10)

        # Info Frame for aligned output
        self.info_frame = tk.Frame(self.main_frame, bg="#F5F5F5")
        self.info_frame.pack(pady=10)

        self.license_plate_label = tk.Label(self.info_frame, text="License Plate: ", font=("Helvetica", 14),
                                            bg="#F5F5F5", justify="center")
        self.license_plate_label.pack(anchor="center")

        self.car_details_label = tk.Label(self.info_frame, text="Car Details: ", font=("Helvetica", 14),
                                          bg="#F5F5F5", justify="center")
        self.car_details_label.pack(anchor="center")

        self.shortest_route_label = tk.Label(self.info_frame, text="", font=("Helvetica", 16, "bold"),
                                             bg="#F5F5F5", justify="center")
        self.shortest_route_label.pack(anchor="center")

        self.capture_button = tk.Button(self.main_frame, text="Capture Image", command=self.capture_image,
                                        font=("Helvetica", 12), bg="#4CAF50", fg="white", padx=10, pady=5)
        self.capture_button.pack(side=tk.LEFT, padx=5)

        self.file_button = tk.Button(self.main_frame, text="Select Image File", command=self.select_image_file,
                                     font=("Helvetica", 12), bg="#FFC107", fg="black", padx=10, pady=5)
        self.file_button.pack(side=tk.LEFT, padx=5)

        self.start_end_button = tk.Button(self.main_frame, text="Input Start/End Points", command=self.input_start_end_points,
                                          font=("Helvetica", 12), bg="#3498DB", fg="white", padx=10, pady=5)
        self.start_end_button.pack(side=tk.LEFT, padx=5)

    def capture_image(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            messagebox.showerror("Error", "Cannot open camera")
            return

        ret, frame = cap.read()
        cap.release()
        if not ret:
            messagebox.showerror("Error", "Failed to capture image")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        license_plate_text, car_details = self.process_image(gray)

        processed_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        photo = ImageTk.PhotoImage(image=Image.fromarray(processed_img))
        self.processed_image_label.configure(image=photo)
        self.processed_image_label.image = photo

        self.license_plate_label.configure(text=f"License Plate: {license_plate_text or 'Not Detected'}")
        self.car_details_label.configure(text=f"Car Details: {car_details or 'Not Found'}")
        self.find_and_display_shortest_route()

    def select_image_file(self):
        file_path = filedialog.askopenfilename(title="Select Image File", filetypes=[("Image Files", ".jpg;.jpeg;*.png")])
        if file_path:
            img = cv2.imread(file_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            license_plate_text, car_details = self.process_image(gray)

            processed_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
            photo = ImageTk.PhotoImage(image=Image.fromarray(processed_img))
            self.processed_image_label.configure(image=photo)
            self.processed_image_label.image = photo

            self.license_plate_label.configure(text=f"License Plate: {license_plate_text or 'Not Detected'}")
            self.car_details_label.configure(text=f"Car Details: {car_details or 'Not Found'}")
            self.find_and_display_shortest_route()

    def process_image(self, img):
        bfilter = cv2.bilateralFilter(img, 11, 17, 17)
        edged = cv2.Canny(bfilter, 30, 200)

        keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(keypoints)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
        location = None

        for contour in contours:
            approx = cv2.approxPolyDP(contour, 10, True)
            if len(approx) == 4:
                location = approx
                break

        if location is None:
            return None, None

        mask = np.zeros(img.shape, np.uint8)
        new_image = cv2.drawContours(mask, [location], 0, 255, -1)
        new_image = cv2.bitwise_and(img, img, mask=mask)

        (x, y) = np.where(mask == 255)
        (x1, y1) = (np.min(x), np.min(y))
        (x2, y2) = (np.max(x), np.max(y))
        cropped_image = img[x1:x2 + 1, y1:y2 + 1]

        reader = easyocr.Reader(['en'])
        result = reader.readtext(cropped_image)

        try:
            text = result[0][-2]
            print("OCR detected:", text)
        except IndexError:
            print("License plate not detected")
            return None, None

        return text, self.find_car_details(text)

    def find_car_details(self, license_plate):
        cleaned_plate = license_plate.replace(" ", "").upper()
        for car in self.cars:
            car_plate = car["Car_Number"].replace(" ", "").upper()
            if cleaned_plate == car_plate:
                details = f"Age: {car['Age_Of_Car']} years, Months without Servicing: {car['Months_without_Servicing']}"
                if int(car["Age_Of_Car"]) > 10:
                    details += "\nWarning: Car is too old and may harm the environment!"
                if int(car["Months_without_Servicing"]) > 6:
                    details += "\nஏய் நண்பா, சேவை செய் (Service the car!)"
                return details
        return "Car details not found."

    def input_start_end_points(self):
        input_window = tk.Toplevel(self.root)
        input_window.title("Input Start/End Points")
        input_window.configure(bg="#F5F5F5")

        start_label = tk.Label(input_window, text="Enter starting point:", font=("Helvetica", 12), bg="#F5F5F5")
        start_label.grid(row=0, column=0, pady=5)
        self.start_entry = tk.Entry(input_window)
        self.start_entry.grid(row=0, column=1, pady=5)

        end_label = tk.Label(input_window, text="Enter destination point:", font=("Helvetica", 12), bg="#F5F5F5")
        end_label.grid(row=1, column=0, pady=5)
        self.end_entry = tk.Entry(input_window)
        self.end_entry.grid(row=1, column=1, pady=5)

        submit_button = tk.Button(input_window, text="Submit", command=lambda: self.submit_start_end_points(input_window),
                                  font=("Helvetica", 12), bg="#4CAF50", fg="white", padx=10, pady=5)
        submit_button.grid(row=2, columnspan=2, pady=10)

    def submit_start_end_points(self, window):
        self.start_point = self.start_entry.get()
        self.end_point = self.end_entry.get()
        window.destroy()
        self.find_and_display_shortest_route()

    def find_and_display_shortest_route(self):
        if self.start_point and self.end_point:
            start = self.start_point.strip().title()
            end = self.end_point.strip().title()
            distance = self.routes.get((start, end))
            if distance:
                self.shortest_route_result = f"Distance from {start} to {end}: {distance} km"
            else:
                self.shortest_route_result = f"No direct route found between {start} and {end}."
            self.shortest_route_label.configure(text=self.shortest_route_result)


if __name__ == "__main__":
    root = tk.Tk()
    app = LicensePlateRecognitionApp(root)
    root.mainloop()
