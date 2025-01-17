import tkinter as tk
from tkinter import ttk, scrolledtext
import cv2
import torch
from PIL import Image, ImageTk
from gtts import gTTS
import os
import threading
from googletrans import Translator
from datetime import datetime
import json
from tkinter import messagebox
import pygame
import time

class ObjectDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Object Detection Speaker")
        self.root.geometry("1024x768")
        pygame.mixer.init()
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.translator = Translator()
        self.current_language = 'en'
        self.cap = None
        self.detection_active = False
        self.viewed_objects = []
        self.last_spoken_time = {}
        self.detection_history_file = "detection_history.json"
        self.audio_lock = threading.Lock()
        self.languages = {
            'English': 'en',
            'Spanish': 'es',
            'French': 'fr',
            'German': 'de',
            'Italian': 'it',
            'Japanese': 'ja'
        }
        self.load_detection_history()
        
        self.setup_gui()

    def setup_gui(self):
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.control_frame = ttk.LabelFrame(self.main_container, text="Controls")
        self.control_frame.pack(fill=tk.X, padx=5, pady=5)
        self.camera_btn = ttk.Button(self.control_frame, text="Open Camera", command=self.toggle_camera)
        self.camera_btn.pack(side=tk.LEFT, padx=5, pady=5)
        self.history_btn = ttk.Button(self.control_frame, text="View Full History", command=self.show_detection_history)
        self.history_btn.pack(side=tk.LEFT, padx=5, pady=5)
        self.clear_btn = ttk.Button(self.control_frame,text="Clear History",command=self.clear_history)
        self.clear_btn.pack(side=tk.LEFT, padx=5, pady=5)
        self.lang_frame = ttk.LabelFrame(self.control_frame, text="Select Language")
        self.lang_frame.pack(side=tk.LEFT, padx=5, pady=5)
        self.language_var = tk.StringVar(value='English')
        self.language_menu = ttk.Combobox(self.lang_frame,textvariable=self.language_var,values=list(self.languages.keys()),state='readonly',width=15)
        self.language_menu.pack(padx=5, pady=5)
        self.language_menu.bind('<<ComboboxSelected>>', self.on_language_change)
        self.middle_frame = ttk.Frame(self.main_container)
        self.middle_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.video_frame = ttk.LabelFrame(self.middle_frame, text="Camera Feed")
        self.video_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        self.video_label = ttk.Label(self.video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.log_frame = ttk.LabelFrame(self.middle_frame, text="Detection Log")
        self.log_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=5)
        self.log_text = scrolledtext.ScrolledText(self.log_frame,height=20,width=40)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def on_language_change(self, event):
        selected_language = self.language_var.get()
        lang_code = self.languages[selected_language]
        self.change_language(lang_code)

    def toggle_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                messagebox.showerror("Error", "Could not open camera")
                return
            
            self.camera_btn.configure(text="Close Camera")
            self.detection_active = True
            threading.Thread(target=self.update_frame, daemon=True).start()
        else:
            self.detection_active = False
            self.cap.release()
            self.cap = None
            self.camera_btn.configure(text="Open Camera")
            self.video_label.configure(image='')
            self.save_detection_history()

    def update_frame(self):
        while self.detection_active:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, (640, 480))
                results = self.model(frame)

                annotated_frame = results.render()[0]
                detected_objects = results.pandas().xyxy[0]['name'].tolist()
                if detected_objects:
                    self.process_detections(detected_objects)
                img = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img = ImageTk.PhotoImage(image=img)
                self.video_label.configure(image=img)
                self.video_label.image = img 
            
            self.root.update_idletasks() 
    def process_detections(self, objects):
        current_time = datetime.now()
        objects_to_speak = []
        
        for obj in objects:
            if obj not in self.last_spoken_time or \
               (current_time - self.last_spoken_time[obj]).total_seconds() > 3:
                translated_name = self.translator.translate(obj, dest=self.current_language).text
                objects_to_speak.append(translated_name)
                self.last_spoken_time[obj] = current_time
                timestamp = current_time.strftime("%Y-%m-%d %H:%M:%S")
                self.viewed_objects.append({
                    'object': obj,
                    'translated': translated_name,
                    'timestamp': timestamp})
        
        if objects_to_speak:
            threading.Thread(target=self.speak_text,args=(" and ".join(objects_to_speak),),daemon=True).start()
            self.update_log()

    def speak_text(self, text):
        try:
            with self.audio_lock:
                tts = gTTS(text=text, lang=self.current_language)
                temp_file = "temp_audio.mp3"
                tts.save(temp_file)
                pygame.mixer.music.load(temp_file)
                pygame.mixer.music.play()
                while pygame.mixer.music.get_busy():
                    time.sleep(0.1)
                    
                pygame.mixer.music.unload()
                os.remove(temp_file)
        except Exception as e:
            print(f"Error in speech synthesis: {e}")

    def load_detection_history(self):
        try:
            if os.path.exists(self.detection_history_file):
                with open(self.detection_history_file, 'r') as f:
                    self.viewed_objects = json.load(f)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load detection history: {e}")
            self.viewed_objects = []

    def save_detection_history(self):
        try:
            with open(self.detection_history_file, 'w') as f:
                json.dump(self.viewed_objects, f)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save detection history: {e}")

    def change_language(self, lang_code):
        self.current_language = lang_code
        self.update_log()

    def update_log(self):
        self.log_text.delete(1.0, tk.END)
        for item in reversed(self.viewed_objects):
            translated_name = self.translator.translate(item['object'],dest=self.current_language).text
            log_entry = f"[{item['timestamp']}] {translated_name}\n"
            self.log_text.insert(tk.END, log_entry)

    def show_detection_history(self):
        history_window = tk.Toplevel(self.root)
        history_window.title("Detection History")
        history_window.geometry("600x400")
        history_text = scrolledtext.ScrolledText(history_window,height=20,width=60)
        history_text.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        for item in reversed(self.viewed_objects):
            translated_name = self.translator.translate(item['object'],dest=self.current_language).text
            history_text.insert(tk.END, f"[{item['timestamp']}] {translated_name}\n")

    def clear_history(self):
        if messagebox.askyesno("Clear History", "Are you sure you want to clear all detection history?"):
            self.viewed_objects = []
            self.update_log()
            self.save_detection_history()
            messagebox.showinfo("Success", "Detection history cleared successfully")

    def cleanup(self):
        if self.cap is not None:
            self.cap.release()
        pygame.mixer.quit()
        self.save_detection_history()
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.protocol("WM_DELETE_WINDOW", app.cleanup)
    root.mainloop()