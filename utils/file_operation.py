import numpy as np
import os


def file_path(file, dir=""):
    if (dir != ""):
        dir = dir + "/"
    return dir + file + ".csv"


# Nos precisamos trocar as labels para se adequarem
# a funcao de ativacao sigmoid
def extract_data(input_file, directory):

    input_file = file_path(input_file, directory)

    data_matrix = []

    with open(input_file, "r") as f:
        for line in f:
            data_matrix.append(
                    np.array(
                        [float(value.strip().strip("\ufeff"))
                            for value in line.split(",")]
                    )
            )
    return np.array(data_matrix)


# Exportacao de dados para um arquivo de texto
def export_data(filename, data_matrix, dir=""):
    if (dir == ""):
        dir = "exportedParams"
    os.makedirs(dir, exist_ok=True)
    np.savetxt(file_path(filename, dir), data_matrix, delimiter=",")
