import json

with open("eval_results/ablation_results_legal.json", encoding="utf-8") as f:
    data = json.load(f)

print("Confidence threshold (tau_conf):")
for row in data.get("confidence_threshold", []):
    val = row["param_value"]
    if val in [0.5, 0.7, 0.95, 0.99]:
        print(f"\\tau_{{\\mathrm{{conf}}}} & {val:.2f} & {row['routing_accuracy']:.4f} & {row['stage2_rate']:.4f} \\\\")

print("\nClarify threshold (tau_clar) - map to ambiguity_clarify_threshold:")
for row in data.get("ambiguity_clarify_threshold", []):
    val = row["param_value"]
    if val in [0.5, 0.7, 0.95, 0.99]:
        print(f"\\tau_{{\\mathrm{{clar}}}} & {val:.2f} & {row['routing_accuracy']:.4f} & {row['stage2_rate']:.4f} \\\\")
