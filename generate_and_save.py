from generate import generate_text  # Import the function from your generate.py file

# Define a list of prompts to generate content from
prompts = [
    "Once upon a time,",
    "In a distant future,",
    "The universe is vast and full of mysteries,",
    "What is the meaning of life?",
    "Explain the concept of quantum mechanics.",
    "A long time ago in a galaxy far, far away,"
]

# Generate and save content to sample.txt
with open('sample.txt', 'a', encoding='utf-8') as file:  # Specify utf-8 encoding
    for prompt in prompts:
        generated_text = generate_text(prompt)  # Call the function to generate text
        file.write(generated_text + "\n\n")  # Write the generated text with extra line breaks

