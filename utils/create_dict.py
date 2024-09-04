
from ast import literal_eval
import utils.file_operation as fop


def extract_dict(dir, filename):
    dictionary = {}

    with open(f"{dir}/{filename}", "r") as f:
        for line in f:
            key, value = line.split()
            dictionary[key] = int(value)

    return dictionary


def extract_tup_dict(dir, filename):
    dictionary = {}

    with open(f"{dir}/{filename}", "r") as f:
        for line in f:
            key, value = line.split()
            if ',' in value:
                dictionary[key] = literal_eval(value)
            elif value.isalpha():
                dictionary[key] = value
            else:
                dictionary[key] = int(value)

    return dictionary


if __name__ == "__main__":
    print(extract_dict("exportedParams", "hyper_param.txt"))
