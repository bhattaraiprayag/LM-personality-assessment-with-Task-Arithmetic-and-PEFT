# experiment_config.py
"""
Configuration file for the experiment pipeline.
"""

### Personality Assessment Parameters: OCEAN
sample_question_bfi10 = "I see myself as someone who "
possible_answers_bfi10 = [
    "is reserved.",
    "is generally trusting.",
    "tends to be lazy.",
    "is relaxed, handles stress well.",
    "has few artistic interests.",
    "is outgoing, sociable.",
    "tends to find fault with others.",
    "does a thorough job.",
    "gets nervous easily.",
    "has an active imagination.",
]
OCEAN_TRAITS = {
    "O": "Openness",
    "C": "Conscientiousness",
    "E": "Extraversion",
    "A": "Agreeableness",
    "N": "Neuroticism",
}
OCEAN_TRAIT_ANSWER_KEYS = {
    "O (+)": "has an active imagination.",
    "O (-)": "has few artistic interests.",
    "C (+)": "does a thorough job.",
    "C (-)": "tends to be lazy.",
    "E (+)": "is outgoing, sociable.",
    "E (-)": "is reserved.",
    "A (+)": "is generally trusting.",
    "A (-)": "tends to find fault with others.",
    "N (+)": "gets nervous easily.",
    "N (-)": "is relaxed, handles stress well.",
}

### Emotion Assessment Parameters: PANAS-X
sample_question_panas_x = "Right now, I feel "
sample_answers_panas_x = [
    "angry.",
    "hostile.",
    "irritable.",
    "scornful.",
    "disgusted.",
    "loathing.",
    "angry at self.",
    "disgusted with self.",
    "sad.",
    "blue.",
    "downhearted.",
    "alone.",
    "lonely.",
    "happy.",
    "joyful.",
    "delighted.",
    "cheerful.",
    "excited.",
    "enthusiastic.",
    "lively.",
    "energetic.",
    "inspired.",
    "proud.",
    "determined.",
    "confident.",
    "bold.",
    "daring.",
    "fearless.",
    "strong.",
]
PANAS_X_TRAIT_SUBCLASSES = {
    "ANGER": [
        "angry",
        "hostile",
        "irritable",
        "scornful",
        "disgusted",
        "loathing",
        "angry at self",
        "disgusted with self",
    ],
    "SADNESS": [
        "sad",
        "blue",
        "downhearted",
        "alone",
        "lonely",
    ],
    "JOY": [
        "happy",
        "joyful",
        "delighted",
        "cheerful",
        "excited",
        "enthusiastic",
        "lively",
        "energetic",
        "inspired",
        "proud",
    ],
    "OPTIMISM": [
        "determined",
        "confident",
        "bold",
        "daring",
        "fearless",
        "strong",
    ],
}


# temperatures = [0.6, 0.9]  # 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1
temperatures = [0.6, 0.7, 0.8, 0.9, 1]  # 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1
# peft_scales = [-12.5, 1.0, 12.5]   # [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 20.0, 25.0, 40.0, 50.0, 75.0, 100.0, 150.0, 200.0] (if using PEFT)
peft_scales = [0.1, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 20.0, 25.0, 40.0, 50.0, 75.0, 100.0, 150.0, 200.0]   # [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 20.0, 25.0, 40.0, 50.0, 75.0, 100.0, 150.0, 200.0] (if using PEFT)
