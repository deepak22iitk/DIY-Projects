import speech_recognition as sr
import pyttsx3

def listen_to_user():
    """
    Captures voice from the default microphone and converts it to text.
    """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        try:
            # Adjust for ambient noise and capture audio
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source, timeout=5)
            print("Recognizing...")
            user_input = recognizer.recognize_google(audio)
            print(f"User said: {user_input}")
            return user_input
        except sr.UnknownValueError:
            print("Sorry, I did not understand.")
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
        except Exception as e:
            print(f"Error: {e}")
        return None

def generate_response(user_input):
    """
    Processes the user input and generates a response.
    """
    # A simple chatbot-like response mechanism
    if user_input:
        if "hello" in user_input.lower():
            return "Hello! How can I assist you today?"
        elif "your name" in user_input.lower():
            return "I am your virtual assistant,Deepak"
        elif "time" in user_input.lower():
            from datetime import datetime
            return f"The current time is {datetime.now().strftime('%H:%M:%S')}."
        else:
            return "Sorry, I don't have an answer for that."
    return "I didn't catch that. Please try again."

def speak_response(response_text):
    """
    Converts text to speech and plays it through the default speaker.
    """
    engine = pyttsx3.init()
    engine.say(response_text)
    engine.runAndWait()

if __name__ == "__main__":
    while True:
        print("Say 'exit' to stop the assistant.")
        user_input = listen_to_user()
        if user_input and "exit" in user_input.lower():
            print("Exiting... Goodbye!")
            speak_response("Goodbye! Have a nice day!")
            break
        
        response = generate_response(user_input)
        print(f"Assistant: {response}")
        speak_response(response)
