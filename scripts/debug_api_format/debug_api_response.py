#!/usr/bin/env python3
"""
Debug script to see the EXACT format of OpenAI Responses API response
Saves raw response to JSON file for analysis
"""

import json
import os
from pathlib import Path
from openai import OpenAI
import pprint
import time

# Load environment variables
env_file = Path(".env")
if env_file.exists():
    with open(env_file) as f:
        for line in f:
            if line.strip() and not line.startswith("#"):
                if "=" in line:
                    key, value = line.strip().split("=", 1)
                    os.environ[key] = value

# Initialize client
client = OpenAI()

# Test with gpt-4.1-mini (our default model)
test_cases = [
    {
        "name": "simple_fact",
        "question": "What is the capital of France?",
        "model": "gpt-4.1-mini",
        "instructions": "You are a helpful assistant. Be concise.",
    },
    {
        "name": "p_vs_np_crypto", 
        "question": "What are the implications of P vs NP for modern cryptography? Provide concrete examples and caveats.",
        "model": "gpt-4.1-mini",
        "instructions": "You are a precise cryptography expert. Be concise and accurate.",
    }
]

for test in test_cases:
    print(f"\n{'='*60}")
    print(f"Test: {test['name']}")
    print(f"Model: {test['model']}")
    print(f"Question: {test['question']}")
    print('='*60)
    
    # Make the API call with logprobs and measure time
    print("\n⏱️  Starting API call...")
    start_time = time.perf_counter()
    
    response = client.responses.create(
        model=test['model'],
        instructions=test.get('instructions', "You are a helpful assistant. Be concise."),
        input=test['question'],
        temperature=0.2,
        top_logprobs=5,
        include=["message.output_text.logprobs"]
    )
    
    end_time = time.perf_counter()
    api_time = end_time - start_time
    print(f"✅ API call completed in {api_time:.2f} seconds")
    
    # Get the raw response in different formats
    print("\n1. Response type:", type(response))
    print("\n2. Response attributes:", dir(response))
    
    # Try to get dictionary representation
    try:
        response_dict = response.model_dump()
        print("\n3. Successfully got model_dump()")
    except Exception as e:
        print(f"\n3. model_dump() failed: {e}")
        try:
            response_dict = json.loads(response.json())
            print("   Fell back to json.loads(response.json())")
        except Exception as e2:
            print(f"   json() also failed: {e2}")
            response_dict = {"error": "Could not serialize response"}
    
    # Save to file
    output_file = f"debug_response_{test['name']}.json"
    with open(output_file, 'w') as f:
        json.dump(response_dict, f, indent=2, default=str)
    print(f"\n4. Saved full response to: {output_file}")
    
    # Analyze the structure
    print("\n5. Response structure analysis:")
    print(f"   - Top level keys: {list(response_dict.keys())}")
    
    if "output" in response_dict:
        print(f"   - output type: {type(response_dict['output'])}")
        if isinstance(response_dict['output'], list):
            print(f"   - output length: {len(response_dict['output'])}")
            for i, item in enumerate(response_dict['output'][:2]):  # First 2 items
                print(f"   - output[{i}] keys: {list(item.keys()) if isinstance(item, dict) else type(item)}")
                if isinstance(item, dict) and 'content' in item:
                    print(f"     - content type: {type(item['content'])}")
                    if isinstance(item['content'], list) and item['content']:
                        print(f"     - content[0] keys: {list(item['content'][0].keys()) if isinstance(item['content'][0], dict) else type(item['content'][0])}")
    
    # Try to extract logprobs specifically
    print("\n6. Logprobs extraction attempt:")
    outputs = response_dict.get("output") or response_dict.get("outputs") or []
    if isinstance(outputs, list):
        for item in outputs:
            if isinstance(item, dict) and item.get("type") == "message":
                for content in item.get("content", []):
                    if isinstance(content, dict):
                        # Check direct logprobs
                        if "logprobs" in content:
                            print(f"   - Found logprobs at content level")
                            print(f"     Keys: {list(content['logprobs'].keys()) if isinstance(content['logprobs'], dict) else type(content['logprobs'])}")
                        # Check nested in output_text
                        if "output_text" in content and isinstance(content["output_text"], dict):
                            if "logprobs" in content["output_text"]:
                                print(f"   - Found logprobs in output_text")
                                lp = content["output_text"]["logprobs"]
                                print(f"     Keys: {list(lp.keys()) if isinstance(lp, dict) else type(lp)}")
                                if isinstance(lp, dict):
                                    if "token_logprobs" in lp:
                                        token_lps = lp["token_logprobs"]
                                        print(f"     token_logprobs type: {type(token_lps)}")
                                        if isinstance(token_lps, list) and token_lps:
                                            print(f"     Sample token_logprobs: {token_lps[:5]}")
                                    if "top_logprobs" in lp:
                                        top_lps = lp["top_logprobs"]
                                        print(f"     top_logprobs type: {type(top_lps)}")
                                        if isinstance(top_lps, list) and top_lps:
                                            print(f"     Sample top_logprobs[0]: {top_lps[0] if top_lps else 'empty'}")
    
    # Extract text
    print("\n7. Text extraction:")
    text = getattr(response, "output_text", None)
    if text:
        print(f"   - Got text from response.output_text: '{text[:100]}...'")
        print(f"   - Full text length: {len(text)} characters")
    else:
        print("   - No direct output_text attribute")
        # Try extraction from dict
        for item in outputs:
            if isinstance(item, dict) and item.get("type") == "message":
                for content in item.get("content", []):
                    if isinstance(content, dict):
                        if "output_text" in content and isinstance(content["output_text"], dict):
                            text = content["output_text"].get("text", "")
                            if text:
                                print(f"   - Extracted from nested structure: '{text[:100]}...'")
                                break
                        elif "text" in content:
                            text = content.get("text", "")
                            if text:
                                print(f"   - Extracted from content.text: '{text[:100]}...'")
                                print(f"   - Full text length: {len(text)} characters")
                                break
    
    # Extract usage info
    print("\n8. Token usage:")
    usage = response_dict.get("usage", {})
    if usage:
        print(f"   - Input tokens: {usage.get('input_tokens', 'N/A')}")
        print(f"   - Output tokens: {usage.get('output_tokens', 'N/A')}")
        print(f"   - Total tokens: {usage.get('total_tokens', 'N/A')}")
    
    # Count logprobs tokens
    logprob_count = 0
    if isinstance(outputs, list):
        for item in outputs:
            if isinstance(item, dict) and item.get("type") == "message":
                for content in item.get("content", []):
                    if isinstance(content, dict) and "logprobs" in content:
                        logprob_count = len(content["logprobs"])
    print(f"   - Logprob tokens found: {logprob_count}")

print("\n" + "="*60)
print("✅ Debug complete! Check the JSON files for full response structure.")
print("="*60)
