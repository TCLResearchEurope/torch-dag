import os
import json
import logging


logger = logging.getLogger(__name__)


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


class JsonResultRegistry:
    def __init__(self, file_path) -> None:
        self.file_path = file_path
        self.ensure_file()

    def ensure_file(self):
        if not os.path.isfile(self.file_path):
            self.make_empty()

    @classmethod
    def default(cls):
        return cls("model_test_results.json")

    def make_empty(self):
        self.write({})

    def read(self):
        with open(self.file_path, "r") as f:
            return json.load(f)

    def _write(self, data):
        with open(self.file_path, "w+") as f:
            json.dump(data, f, indent=4, sort_keys=True)
            
    def write(self, data):
        try:
            self._write(data)
        except KeyboardInterrupt:
            logger.warning("Interrupted by user during writing to file, attempting to finish the write to avoid corrupted results.")
            self._write(data)
            raise KeyboardInterrupt

    def update(self, model_name: str, entry: dict):
        data = self.read()
        if entry:
            if model_name not in data:
                data[model_name] = {}
            data[model_name].update(**entry)
            self.write(data)
