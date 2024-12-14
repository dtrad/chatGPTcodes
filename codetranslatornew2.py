#! /usr/bin/env python
# define is a python file using the environment in the current directory
# to translate code from one programming language to another using OpenAI's updated API.


import openai
from openai import OpenAI
import os
import sys
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
import os
# Set your OpenAI API key
# You can replace os.environ.get("OPENAI_API_KEY") with your actual key for testing, but do not hardcode it in production.

def translate_code(input_code, source_language, target_language):
    """
    Translates code from one programming language to another using OpenAI's updated API.

    Args:
        input_code (str): The code to translate.
        source_language (str): The language of the input code (e.g., 'Python').
        target_language (str): The desired output language (e.g., 'JavaScript').

    Returns:
        str: Translated code.
    """
    prompt = (
        f"Translate the following {source_language} code into {target_language}:\n\n"
        f"{input_code}\n\n"
        "Ensure that the translation is accurate, idiomatic, and uses best practices for the target language."
    )

    try:
        response = client.chat.completions.create(model="gpt-4",  # Ensure you're using the correct model
        messages=[
            {"role": "system", "content": "You are an expert coding assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2)
        # Extract the assistant's reply
        translated_code = response.choices[0].message.content.strip()
        return translated_code
    except openai.OpenAIError as e:
        return f"OpenAI API Error: {e}"
    except Exception as e:
        return f"Error: {e}"

# define a function to eliminate from the output all lines before and after "```" and the "```" itself
def clean_output(output):
    output = output.split("```")
    if len(output) > 1:
        output = output[1]
    else:
        output = output[0]
    return output

# remove first line of the output
def remove_first_line(output):
    output = output.split("\n", 1)[1]
    return output
        

# read the code from a file given as the first argument in command line, and translate it from Python to the target language in the second argument
# use as output name the third argument
import sys
if len(sys.argv) < 5:
    print("Usage: python codetranslatornew2.py <input_file> <input_language> <target_language> <output_file>")
    sys.exit()
inputlang = sys.argv[2]
outputlang = sys.argv[3]

with open(sys.argv[1], 'r') as file:
    code = file.read()
    translated_code = translate_code(code, inputlang, outputlang)
    translated_code2 = clean_output(translated_code)
    # remove the first line before writing to the output file
    

    with open(sys.argv[4], 'w') as output_file:
        # remove the first line before writing to the output file
        translated_code3 = remove_first_line(translated_code2)
        output_file.write(translated_code3)
        

