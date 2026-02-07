# remedy_generator.py

import os
from dotenv import load_dotenv
from openai import OpenAI

# Load .env file
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_remedy(disease_name):
    try:
        prompt = f"Suggest effective natural or chemical remedies for {disease_name} in plants."

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a plant pathologist."},
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print("❌ Failed to generate remedy:\n", e)
        return "Sorry, couldn't generate a remedy."

