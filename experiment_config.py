# experiment_config.py
"""
Configuration file for the experiment pipeline.
"""

temperatures = [0.6, 0.7, 0.8, 0.9, 1]  # 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1
sample_question = "I see myself as someone who"
possible_answers = [
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
peft_scales = [0.1, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 20.0, 25.0, 40.0, 50.0, 75.0, 100.0, 150.0, 200.0]   # [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 20.0, 25.0, 40.0, 50.0, 75.0, 100.0, 150.0, 200.0] (if using PEFT)
