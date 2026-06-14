import csv
import os
from config import Config

def log(iteration, loss, elo, games_played):
    os.makedirs(os.path.dirname(Config.log_path), exist_ok=True)
    file_exists = os.path.isfile(Config.log_path)
    
    with open(Config.log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['iteration','loss','elo','games_played'])
          
        writer.writerow([iteration, round(loss, 4), elo, games_played])
