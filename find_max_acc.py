import json
from pathlib import Path

path = Path(r"a:\Programming\School\cs565\project\technical\evaluation_results\full_results.json")

try:
    with open(path, 'r') as f:
        data = json.load(f)
    
    results = []
    if 'verification' in data:
        for res, methods in data['verification'].items():
            for method, metrics in methods.items():
                if isinstance(metrics, dict) and 'accuracy' in metrics:
                    results.append((res, method, metrics['accuracy']))
    
    results.sort(key=lambda x: x[2], reverse=True)
    
    for res, method, acc in results:
        print(f"{res} - {method}: {acc}")

except Exception as e:
    print(f"Error: {e}")
