
# Test actual Anthropic API execution without modifying existing NLP code

import os
import isl_nlp

print("MEMBER 3 - LLM EXECUTION TEST")
print("-" * 40)

# Check API key without displaying its value
if not os.getenv("ANTHROPIC_API_KEY"):
    print("ERROR: ANTHROPIC_API_KEY is not configured.")
else:
    print("API key found.")

# Check configured model
print("Configured model:", isl_nlp.LLM_MODEL)

# Test actual LLM execution
result = isl_nlp._llm(
    "Reply with exactly: API_WORKING",
    max_tokens=30
)

if result:
    print("LLM RESPONSE:", result)
    print("STATUS: LLM EXECUTION SUCCESSFUL")
else:
    print("STATUS: LLM EXECUTION FAILED")
    print("Check the API key, model ID and API access.")
