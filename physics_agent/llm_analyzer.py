import json
import openai  # Assuming LLM provider like OpenAI

openai.api_key = 'your-api-key'  # Set securely

def analyze_sweep_results(sweep_results: list) -> dict:
    """Use LLM to pick best sweep params based on results."""
    prompt = "Analyze these sweep results and pick the best parameters. Prioritize beating SOTA in PTA/CMB, avoid disqualified ones. Results: " + json.dumps(sweep_results, default=str)
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "You are a physics expert selecting optimal sweep parameters."}, {"role": "user", "content": prompt}]
    )
    best_params = json.loads(response.choices[0].message.content)  # Assume LLM outputs JSON
    return best_params 

def recommend_sweep_values(theory, field: str, past_results: list) -> list:
    """LLM recommends sweep values for a field based on theory and past results."""
    prompt = f"For {theory.name}, recommend {field} sweep values (list of 5-10 floats) to optimize PTA/CMB. Past results: {json.dumps(past_results, default=str)}. Avoid disqualified."
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "system", "content": "Recommend optimal sweep values as JSON list."}, {"role": "user", "content": prompt}]
    )
    return json.loads(response.choices[0].message.content)  # Returns e.g., [0.1,0.2,0.3,0.4,0.5] 