import json
from json import JSONEncoder
import numpy as np


class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyArrayEncoder, self).default(obj)


def dump(filename, np_data):
    with open(filename, "w") as write_file:
        json.dump(np_data, write_file, cls=NumpyArrayEncoder)


def loads(filename):
    with open(filename, "r") as read_file:
        decoded_data = json.load(read_file)

    return decoded_data
