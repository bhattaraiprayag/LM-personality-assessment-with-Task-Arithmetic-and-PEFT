# experiment_config.py
"""
Configuration file for the experiment pipeline.
"""
from pathlib import Path


# BFI-10
BFI10_ITEMS = [
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
BFI10_LIKERT = [
    "Disagree strongly",
    "Disagree a little",
    "Neither agree nor disagree",
    "Agree a little",
    "Agree strongly",
]
BFI10_TRAIT_POLARITY = {
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


# PANAS‑X
PANASX_ITEMS = [
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
PANASX_LIKERT = [
    "Very slightly or not at all",
    "A little",
    "Moderately",
    "Quite a bit",
    "Extremely",
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


# IPIP‑120
IPIP_ITEMS_FILE = Path("data/ipip120_items.csv")
IPIP_LIKERT = [
    "Very inaccurate",
    "Moderately inaccurate",
    "Neither accurate nor inaccurate",
    "Moderately accurate",
    "Very accurate",
]


# MAIN INVENTORY CONFIG
INVENTORIES: dict[str, dict] = {
    "BFI10": {
        "question_stem": "How well does the following statement describe you?",
        "statement_template": "STATEMENT: I see myself as someone who {item}",
        "anchors": BFI10_LIKERT,
        "items": BFI10_ITEMS,
    },
    "PANASX": {
        "question_stem": "Indicate to what extent this statement describes how you feel.",
        "statement_template": "STATEMENT: Right now, I feel {item}",
        "anchors": PANASX_LIKERT,
        "items": PANASX_ITEMS,
    },
    "IPIP120": {
        "question_stem": (
            "The following statement describes people's behaviours. Please select how accurately "
            "this statement describes you.\n\nDescribe yourself as you generally are now, not as you "
            "wish to be in the future. Describe yourself as you honestly see yourself, in relation "
            "to other people you know of the same sex and roughly the same age."
        ),
        "statement_template": "STATEMENT: I {item}",
        "anchors": IPIP_LIKERT,
        "items_file": IPIP_ITEMS_FILE   # ipip120_items.csv
    },
}


# EXPERIMENT CONFIGS
temperatures = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
# temperatures = [0, 0.1]
peft_scales = [-25.0, -20.0, -15.0, -12.5, -10.0, -7.5, -5.0, -2.5, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5, 1.0, 2.5, 5.0, 7.5, 10.0, 12.5, 15.0, 20.0, 25.0]
# peft_scales = [-12.5, 0.1, 0.2]
