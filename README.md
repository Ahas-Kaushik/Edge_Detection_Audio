# Edge_Detection_Audio
Detects objects and identifies them and OP's it via Audio


Ignore Imports if not necessary.

Imports!!!!
Libraries Required!!!!
pip install torch torchvision
pip install ultralytics
pip install opencv-python
pip install pillow
pip install gtts
pip install playsound
pip install googletrans==3.1.0a0
pip install chardet==4.0.0

Work Flow of the functions:
1. setup_gui:
Creates the GUI layout with buttons for camera toggle, viewing/clearing history, and changing languages, as well as frames for video display and logs.

2. toggle_camera:
Starts/stops the camera feed and updates the button text accordingly. If stopping, saves the detection history.

3. update_frame:
Captures video frames, performs object detection, overlays results, and updates the GUI video feed. Detected objects are processed.

4. process_detections:
Handles detected objects by translating names, updating timestamps, and triggering text-to-speech for objects detected after a cooldown period.

6. speak_text:
Converts text to speech using gTTS and plays the audio.

7. change_language:
Updates the current language for translation and refreshes the detection log.

8. update_log:
Updates the on-screen detection log with translated object names and timestamps.

9. save_detection_history and load_detection_history:
Save and load detection history from a JSON file for persistence.

10. show_detection_history:
Displays detection history in a separate window.

11. clear_history:
Clears the detection history and updates the log.

This Model is not fully absolute or perfectly accurate it still needs some upgrades and somemore fine tunes to get the models accuracy and the buffer in the frame rate. 
Thank You.
