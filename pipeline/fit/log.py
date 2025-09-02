from pathlib import Path

class Logger:
    def __init__(self):
        # self.save_folder = None
        pass

    def create_log(self, log_path):
        self.log_path = log_path
        with open(self.log_path, 'a'):
            pass
    
    def print(self, text):
        with open(self.log_path, 'a') as f:
            f.write(text)

logger = Logger()

