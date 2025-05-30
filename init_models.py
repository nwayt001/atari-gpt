from openai import OpenAI 
import anthropic

def init_model(gemini=False, claude=False):

    if gemini:
        with open("GEMINI_API_KEY.txt", "r") as gemini_api_file: 
            gemini_api_file = gemini_api_file.read().strip() 
        client = OpenAI(api_key=gemini_api_file,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

    elif claude:
        with open("ANTHROPIC_API_KEY.txt", "r") as claude_api_file: 
            claude_api_file = claude_api_file.read().strip() 
        client = anthropic.Anthropic(api_key=claude_api_file)
    else:
        with open("OPENAI_API_KEY.txt", "r") as openai_api_file: 
            openai_api_key = openai_api_file.read().strip() 
        client = OpenAI(api_key=openai_api_key)

    return client