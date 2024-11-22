import json
import random

# Sample topics and templates for generating prompts and responses
topics = [
    "Politics", "Sports", "Technology", "Science", "Entertainment", "Business", "Health", "Education"
]

questions = [
    "What is the latest update on {topic}?",
    "Can you summarize the key points in {topic}?",
    "What are the implications of recent developments in {topic}?",
    "How has {topic} been evolving over the past week?",
    "What are the top headlines in {topic} today?"
]

responses = [
    "The latest in {topic} shows significant changes, including {details}.",
    "Recent developments in {topic} include {details}.",
    "Key highlights in {topic} are {details}.",
    "{topic} is seeing a shift due to {details}.",
    "Experts are discussing the impact of {details} on {topic}."
]

# Function to generate random details for responses
def generate_details(topic):
    keywords = ["policy changes", "technological breakthroughs", "market trends", "scientific discoveries"]
    return f"{random.choice(keywords)} related to {topic}."

# Generate synthetic data
def generate_data(file_name, num_samples=50000):
    with open(file_name, "w", encoding="utf-8") as file:
        for _ in range(num_samples):
            topic = random.choice(topics)
            prompt = random.choice(questions).format(topic=topic)
            response = random.choice(responses).format(topic=topic, details=generate_details(topic))
            data = {"prompt": prompt, "response": response}
            file.write(json.dumps(data) + "\n")

# Generate train and test datasets
generate_data("train.jsonl", num_samples=50000)  # Adjust samples to reach ~10 MB
generate_data("test.jsonl", num_samples=10000)   # Smaller test dataset
