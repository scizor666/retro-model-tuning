import json, os
for f in os.listdir("data/eval_results"):
    if f.endswith(".json"):
        d = json.load(open(os.path.join("data/eval_results", f)))
        print(f"{f}: {len(d.get('results', []))} items, Acc: {d.get('overall_accuracy')}")
