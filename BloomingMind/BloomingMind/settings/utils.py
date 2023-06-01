from pathlib import Path
import json
import os

BASE_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = os.path.dirname(BASE_DIR)
from django.core.management.utils import get_random_secret_key


def write_random_secret_key():
    secret_key = get_random_secret_key()
    save_path = os.path.join(ROOT_DIR, "Core/env/env.json")
    data = {
        "SECRET_KEY": secret_key
    }

    with open(save_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)

