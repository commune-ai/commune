import openai
import os

# Ensure your OPENAI_API_KEY is set in your environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_ai_text(prompt: str, temperature: float) -> str:
    """
    Generates text based on the provided prompt using OpenAI's GPT-4 preview model.
    
    :param prompt: The prompt to send to the model.
    :param temperature: The temperature to use for the generation. Lower means more deterministic.
    :return: The generated text as a string.
    """
    try:
        response = openai.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": f"Temperature: {temperature}"},
                {"role": "user", "content": prompt},
            ],
            max_tokens=3000,
            stop=None,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"An error occurred: {e}")
        return "Error generating text."

if __name__ == "__main__":
    # Test the function with a sample prompt
    test_prompt = "Tell me a story about a robot learning to love."
    print(generate_ai_text(test_prompt, 0.7))
