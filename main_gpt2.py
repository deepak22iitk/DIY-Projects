import speech_recognition as sr
import pyttsx3
from transformers import pipeline

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

def generate_response_with_gpt(user_input, chatbot):
    """
    Generates a response using the GPT model.
    """
    if user_input:
        print("Generating response with GPT...")
        response = chatbot(user_input, max_length=150, num_return_sequences=1)
        return response[0]['generated_text']
    return "I didn't catch that. Please try again."

def speak_response(response_text):
    """
    Converts text to speech and plays it through the default speaker.
    """
    engine = pyttsx3.init()
    engine.say(response_text)
    engine.runAndWait()

if __name__ == "__main__":
    # Initialize the Hugging Face pipeline for text generation
    print("Initializing GPT model...")
    chatbot = pipeline("text-generation", model="distilgpt2")  # Replace "gpt2" with a lighter model if needed
    #model:distilgpt2
    while True:
        print("Say 'exit' to stop the assistant.")
        user_input = listen_to_user()
        if user_input and "exit" in user_input.lower():
            print("Exiting... Goodbye!")
            speak_response("Goodbye! Have a nice day!")
            break

        # Generate dynamic response with GPT
        response = generate_response_with_gpt(user_input, chatbot)
        print(f"Assistant: {response}")
        speak_response(response)
