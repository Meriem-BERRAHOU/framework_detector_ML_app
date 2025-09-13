import json
import os
import pandas as pd

def compare_results(results_dir):
    all_results = []

    for file in os.listdir(results_dir):
        if file.endswith(".json"):
            with open(os.path.join(results_dir, file), "r") as f:
                data = json.load(f)
                all_results.append({
                    "model": data["model"],
                    "accuracy": data["accuracy"]
                })

    df = pd.DataFrame(all_results)
    print("\n📊 Comparaison des modèles :")
    print(df.sort_values(by="accuracy", ascending=False))
    return df
  

if __name__ == "__main__":
    results = compare_results("results")
