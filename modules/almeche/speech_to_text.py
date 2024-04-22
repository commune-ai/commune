import speech_recognition as sr

def recognize_speech_from_mic(recognizer, microphone):
    """
    Transcribes speech from recorded from `microphone`.
    
    :param recognizer: The speech recognition recognizer instance.
    :param microphone: The microphone instance to listen for the speech.
    :return: A dictionary with three keys:
             "success": a boolean indicating whether or not the API request was successful;
             "error":   `None` if no error occured, otherwise a string describing the error;
             "transcription": `None` if speech could not be transcribed, otherwise a string with the transcription.
    """
    # Check that recognizer and microphone arguments are appropriate type
    if not isinstance(recognizer, sr.Recognizer):
        raise TypeError("`recognizer` must be `Recognizer` instance")
    if not isinstance(microphone, sr.Microphone):
        raise TypeError("`microphone` must be `Microphone` instance")

    # Adjust the recognizer sensitivity to ambient noise and record audio
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    # Set up the response object
    response = {
        "success": True,
        "error": None,
        "transcription": None
    }

    # Try recognizing the speech in the recording
    try:
        response["transcription"] = recognizer.recognize_google(audio)
    except sr.RequestError:
        # API was unreachable or unresponsive
        response["success"] = False
        response["error"] = "API unavailable"
    except sr.UnknownValueError:
        # Speech was unintelligible
        response["error"] = "Unable to recognize speech"

    return response

def recognize_speech():
    """
    Captures and recognizes speech from the user's microphone.
    
    :return: Transcribed text from the speech.
    """
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()

    print("Please speak your idea for a CAD object:")
    speech = recognize_speech_from_mic(recognizer, microphone)

    if speech["success"]:
        return speech["transcription"]
    else:
        return f"Error: {speech['error']}"

if __name__ == "__main__":
    # Test the function
    print(recognize_speech())
