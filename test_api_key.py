"""Simple script to test if the Google Gemini API key is valid."""

import os
import google.generativeai as genai

# Your API key
api_key = os.environ.get('GOOGLE_API_KEY') or os.environ.get('GEMINI_API_KEY') or ""

print("Testing API key...")
print(f"API key: {api_key[:20]}...")

try:
    # Configure the API
    genai.configure(api_key=api_key)
    
    # Create a model instance
    model = genai.GenerativeModel('gemini-2.5-flash-lite')
    
    # Make a simple test call
    print("\nMaking test API call...")
    response = model.generate_content("Say 'Hello' if you can read this.")
    
    print(f"\n✅ SUCCESS! API key is valid!")
    print(f"Response: {response.text}")
    
except Exception as e:
    error_msg = str(e)
    print(f"\n❌ ERROR: API key test failed")
    print(f"Error: {error_msg}")
    
    if "API_KEY_INVALID" in error_msg or "API Key not found" in error_msg:
        print("\n" + "="*60)
        print("The API key is INVALID or not working.")
        print("\nTo fix this:")
        print("1. Go to: https://makersuite.google.com/app/apikey")
        print("2. Sign in with your Google account")
        print("3. Click 'Create API Key' or check your existing keys")
        print("4. Make sure the 'Generative AI API' is enabled")
        print("5. Copy the new API key and update it in the script")
        print("="*60)
    elif "PERMISSION_DENIED" in error_msg:
        print("\nThe API key doesn't have the right permissions.")
        print("Make sure the 'Generative AI API' is enabled for this key.")
    else:
        print(f"\nUnexpected error: {e}")

