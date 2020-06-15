_NAME_TOKEN = "name"
_METADATA_TOKEN = "metadata"

DISEASE_NAME_TO_COST_DICT = {
    "healthy": 0,
    "cold": 1,
    "flue": 1,
    "coronavirus": 2,
    "lung cancer": 3,
}

DISEASE_NAME_TO_LABEL_DICT = {
    "healthy": 0,
    "cold": 1,
    "flue": 2,
    "coronavirus": 3,
    "lung cancer": 4,
}

assert sorted(DISEASE_NAME_TO_LABEL_DICT.keys()) == sorted(DISEASE_NAME_TO_COST_DICT.keys())

DISEASE_LABEL_TO_NAME_DICT = {v: k for k, v in DISEASE_NAME_TO_LABEL_DICT.items()}
DISEASE_LABEL_TO_COST_DICT = {v: DISEASE_NAME_TO_COST_DICT[k] for k, v in DISEASE_NAME_TO_LABEL_DICT.items()}
