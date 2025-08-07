import os
from dotenv import load_dotenv
from llama_api_client import LlamaAPIClient
# from llama_api_client.types import UserMessageParam

load_dotenv()

client = LlamaAPIClient(
    api_key=os.environ.get("LLAMA_API_KEY")
)

def query_llama(term):
    try:
        # Step 1: Get detailed meaning
        detailed_response = client.chat.completions.create(
            model="Llama-4-Maverick-17B-128E-Instruct-FP8",
            messages=[
                {"role": "user", "content": f"What does the slang term '{term}' mean?"}
            ],
        )
        full_meaning = detailed_response.completion_message.content.text.strip()

        # Step 2: Use summarize_definition for summary
        short_summary = summarize_definition(term, full_meaning)

        return f"{full_meaning}\n\n✏️ Simple Summary: {short_summary}"

    except Exception as e:
        print("LLAMA API Error:", e)
        return None

def summarize_definition(term, definition, summary=None):
    """
    Returns a summary for the slang term.
    If `summary` is provided (e.g., from a dataset), returns it.
    Otherwise, queries the LLM for a summary.
    """
    if summary:
        return summary
    try:
        response = client.chat.completions.create(
            model="Llama-4-Maverick-17B-128E-Instruct-FP8",
            messages=[
                {
                    "role": "user",
                    "content": f"Summarize the slang term '{term}', which means: '{definition}', in one short sentence using simple English."
                }
            ],
        )
        return response.completion_message.content.text.strip()
    except Exception as e:
        print("LLAMA Summary Error:", e)
        return "Summary not available."

# === NEW ENHANCED FUNCTIONS ===

def query_llama_for_translation(text, input_type="auto"):
    """Enhanced function for bidirectional translation"""
    try:
        if input_type == "slang" or input_type == "auto":
            # Try slang to standard translation
            slang_response = client.chat.completions.create(
                model="Llama-4-Maverick-17B-128E-Instruct-FP8",
                messages=[
                    {
                        "role": "user", 
                        "content": f"Translate this slang text to standard English: '{text}'. If it's not slang, just say 'not slang'."
                    }
                ],
            )
            slang_result = slang_response.completion_message.content.text.strip()
            
            if "not slang" not in slang_result.lower():
                return f"**Slang → Standard:** {slang_result}"
        
        if input_type == "standard" or input_type == "auto":
            # Try standard to slang translation
            standard_response = client.chat.completions.create(
                model="Llama-4-Maverick-17B-128E-Instruct-FP8",
                messages=[
                    {
                        "role": "user", 
                        "content": f"Convert this standard English to modern slang/informal language that young people would use: '{text}'. Keep the same meaning but make it more casual and trendy."
                    }
                ],
            )
            standard_result = standard_response.completion_message.content.text.strip()
            return f"**Standard → Slang:** {standard_result}"
        
        return None

    except Exception as e:
        print(f"LLAMA Translation Error: {e}")
        return None

def get_contextual_translation(text, context="", direction="auto"):
    """Get context-aware translation using LLAMA"""
    try:
        context_info = f" (Context: {context})" if context else ""
        
        if direction == "slang_to_standard":
            prompt = f"Translate this slang to formal English{context_info}: '{text}'"
        elif direction == "standard_to_slang":
            prompt = f"Convert this formal English to modern slang{context_info}: '{text}'"
        else:
            prompt = f"Provide both slang-to-standard and standard-to-slang translations{context_info}: '{text}'"
        
        response = client.chat.completions.create(
            model="Llama-4-Maverick-17B-128E-Instruct-FP8",
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        
        return response.completion_message.content.text.strip()
    
    except Exception as e:
        print(f"LLAMA Contextual Translation Error: {e}")
        return None