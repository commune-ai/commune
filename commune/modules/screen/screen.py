

import os
import commune as c

import cv2
import numpy as np
import pyautogui
import threading
import speech_recognition as sr
import time
from datetime import datetime
import tkinter as tk
from tkinter import filedialog

class ScreenRecorder:
    def __init__(self):
        self.recording = False
        self.output_file = None
        self.writer = None
        self.screen_width, self.screen_height = pyautogui.size()
        self.recognizer = sr.Recognizer()
        self.recording_thread = None

    def record_screen(self):
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.writer = cv2.VideoWriter(self.output_file, fourcc, 20.0, 
                                    (self.screen_width, self.screen_height))

        while self.recording:
            # Capture screen
            screen = pyautogui.screenshot()
            frame = np.array(screen)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.writer.write(frame)

        # Release resources
        self.writer.release()

    def listen_for_commands(self):
        with sr.Microphone() as source:
            print("Listening for commands ('m start' to begin, 'm stop' to end)...")
            while True:
                try:
                    audio = self.recognizer.listen(source)
                    command = self.recognizer.recognize_google(audio).lower()
                    print(f"Recognized: {command}")

                    if 'm start' in command and not self.recording:
                        self.start_recording()
                    elif 'm stop' in command and self.recording:
                        self.stop_recording()

                except sr.UnknownValueError:
                    pass
                except sr.RequestError:
                    print("Could not request results; check your internet connection")

    def get_save_location(self):
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"screen_recording_{current_time}.avi"
        file_path = filedialog.asksaveasfilename(
            defaultextension=".avi",
            initialfile=default_filename,
            filetypes=[("AVI files", "*.avi"), ("All files", "*.*")]
        )
        return file_path

    def start_recording(self):
        if not self.recording:
            self.output_file = self.get_save_location()
            if self.output_file:
                print("Starting recording...")
                self.recording = True
                self.recording_thread = threading.Thread(target=self.record_screen)
                self.recording_thread.start()
            else:
                print("Recording cancelled - no save location selected")

    def stop_recording(self):
        if self.recording:
            print("Stopping recording...")
            self.recording = False
            if self.recording_thread:
                self.recording_thread.join()
            print(f"Recording saved to: {self.output_file}")

def main():
    recorder = ScreenRecorder()
    try:
        recorder.listen_for_commands()
    except KeyboardInterrupt:
        if recorder.recording:
            recorder.stop_recording()
        print("\nExiting...")






if __name__ == "__main__":
    main()