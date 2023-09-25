from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain.prompts.prompt import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
import requests
import os

# Load environment variables from a .env file
load_dotenv(find_dotenv())

# Retrieve API tokens from environment variables
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
OPEN_AI_KEY = os.getenv('OPEN_AI_KEY')

# Function to convert an image to text using Hugging Face Transformers pipeline
def img2text(url):
    # Create an image-to-text pipeline using a specific model
    image_to_text = pipeline("image-to-text", model='Salesforce/blip-image-captioning-base')

    # Extract text from the provided image URL
    text = image_to_text(url)[0]['generated_text']

    # Print the extracted text and return it
    print(text)
    return text

# Function to generate a story based on a given scenario
def generate_story(scenario):
    
    ## Define a template for generating a story
    template = """
    You are a story teller;
    You can generate a short story based on a simple narrative, the story should be no more than 20 words;
    
    CONTEXT: {scenario}
    STORY:
    """
    
    prompt = PromptTemplate(template=template, input_variables=["scenario"])
    
    
    story_llm = LLMChain(llm=ChatOpenAI(
        model_name="gpt-3.5-turbo", temperature=1, openai_api_key =OPEN_AI_KEY , verbose=True, max_tokens=2000), prompt=prompt)

    story = story_llm.predict(scenario=scenario)
    print(story)
    return story

#text to speech
def text2speech(message):
    API_URl = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f'Bearer {HUGGINGFACEHUB_API_TOKEN}'}
    payloads = {
        "inputs": message
    }

    response = requests.post(API_URl, headers=headers, json=payloads)
    with open('audio.flac', 'wb') as file:
        file.write(response.content)
    

# Call img2text with a maximum length of 100 tokens
scenario = img2text("we2.JPG")
story = generate_story(scenario)
text2speech(story)