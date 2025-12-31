import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found. Check your .env file.")


client = genai.Client(api_key=api_key)

def run_experiment(
    image_path,
    prompt_text,
    temperature,
    top_p,
    top_k
):
    image = Image.open(image_path)

    from google.genai import types

    response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[image, prompt_text],
            config=types.GenerateContentConfig(
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                max_output_tokens=1500
            )
        )


    return response.text



# TASK 1 — INTERIOR DESIGN

room_prompt = """
Suggest a modern minimalist interior design for this room.
Preferences:
- Warm lighting
- Budget-friendly
- Suitable for a working professional
"""

print("\n--- Interior Design Output ---")
print(run_experiment(
    "images/room.png",
    room_prompt,
    temperature=0.6,
    top_p=0.95,
    top_k=50
))



# TASK 2 — DIAGRAM EXPLANATION

diagram_prompt = """
Explain this neural network diagram to a beginner software engineer.
Describe each layer and how data flows.
"""

print("\n--- Diagram Explanation Output ---")
print(run_experiment(
    "images/neural_network.png",
    diagram_prompt,
    temperature=0.2,
    top_p=0.9,
    top_k=40
))




# TASK 3 — FOOD RECIPE

food_prompt = """
Create a healthy vegetarian recipe inspired by this dish.
Constraints:
- No dairy
- High protein
- Suitable for Indian diet
"""

print("\n--- Recipe Output ---")
print(run_experiment(
    "images/pasta.png",
    food_prompt,
    temperature=0.7,
    top_p=0.95,
    top_k=60
))
