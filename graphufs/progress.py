"""Written by ChatGPT"""
import json
import os

class ProgressTracker:
    def __init__(self, json_file_path):
        self.json_file_path = json_file_path
        dirname = os.path.dirname(json_file_path)
        if not os.path.isdir(dirname):
            os.makedirs(dirname)
        self.current_iteration = 0
        self.load_progress()

    def load_progress(self):
        """Loads the last saved iteration from the JSON file, if it exists."""
        if os.path.exists(self.json_file_path):
            with open(self.json_file_path, 'r') as file:
                data = json.load(file)
                self.current_iteration = data.get("iteration", 0)
        else:
            self.current_iteration = 0

    def update_progress(self, iteration):
        """Updates the JSON file with the current iteration."""
        self.current_iteration = iteration
        with open(self.json_file_path, 'w') as file:
            json.dump({"iteration": self.current_iteration}, file)

    def get_current_iteration(self):
        """Returns the last saved iteration number."""
        return self.current_iteration
