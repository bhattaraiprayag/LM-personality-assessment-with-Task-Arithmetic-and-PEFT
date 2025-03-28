import json

# Load the JSON data
with open('./outputs/experiment_metadata.json', 'r') as file:
    data = json.load(file)

def normalize_probabilities(data):
    # Traverse each experiment
    for experiment, details in data.items():
        results = details.get("results", {})
        
        # Traverse each phase (pre/post)
        for phase_key in ["personality_eval_pre", "personality_eval_post", "custom_eval_pre", "custom_eval_post"]:
            phase_data = results.get(phase_key, {})
            # Traverse each scale
            for scale_key, answers_list in phase_data.items():
                # Group by temperature
                temp_groups = {}
                for entry in answers_list:
                    temp = entry['temp']
                    if temp not in temp_groups:
                        temp_groups[temp] = []
                    temp_groups[temp].append(entry)

                # Normalize within each temperature group
                for temp, group in temp_groups.items():
                    total_prob = sum(item['prob'] for item in group)
                    if total_prob > 0:
                        for item in group:
                            item['prob'] /= total_prob

    return data

# Normalize the probabilities
data_normalized = normalize_probabilities(data)

# Save the normalized data back to a file
with open('./outputs/experiment_metadata_norm.json', 'w') as file:
    json.dump(data_normalized, file, indent=4)
