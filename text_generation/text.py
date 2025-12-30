import os
from google import genai
from google.genai import types
from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
response = client.models.generate_content(
    model = "gemini-3-flash-preview",
    contents = ["""Draft an engaging opening section for a technical blog aimed at software engineers.
    The introduction should clearly explain what Large Language Models (LLMs) are,
    why they matter, and how developers typically interact with them.""",

    """Write a persuasive and user-friendly product description for a modern audio device.
    The product is a pair of wireless noise-cancelling headphones that offer:
    • Up to 40 hours of battery life on a single charge
    • Support for Bluetooth version 5.3
    Highlight benefits, not just features.""",

    """ Create an original science-fiction short story between 300 and 400 words.
    The story must begin with the line:
    "The last human on Earth received a message saying: I'm not alone."
    Structure the narrative with a clear beginning, middle, and conclusion,
    and end with an unexpected twist that redefines the message."""
    ] ,
    config = types.GenerateContentConfig(
        temperature = 0.1,
        top_p = 0.9,
        max_output_tokens = 1500
    )
)
print(response.text)