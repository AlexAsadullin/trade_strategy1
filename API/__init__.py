# libs
import sys
import os
import shutil
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# project dependences
from api_main import app

# clearance of temp directory
current_directory = Path(__file__).resolve().parent
folder_path = os.path.join(current_directory, "temp")
if not os.path.exists(folder_path):
        os.makedirs(folder_path)

for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    os.remove(file_path)

