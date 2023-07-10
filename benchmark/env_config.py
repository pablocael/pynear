"""
Config settings
"""
import os

from dotenv import load_dotenv

load_dotenv()


# the temp folder used to store donwloaded videos
DATASETS_DIR = os.getenv("DATASETS_DIR", "datasets")
