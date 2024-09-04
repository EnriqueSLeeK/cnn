
import numpy as np
import utils.file_operation as fop
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd


class confusion_matrix:
    def __init__(self, label_size):
        self.conf_matrix = np.zeros(shape=(label_size, label_size))

    def update_matrix(self, expected_index, predicted_index):
        for ei, pi in zip(expected_index, predicted_index):
            self.conf_matrix[ei][pi] += 1

    def export_matrix(self):
        fop.export_data("confusion_matrix", self.conf_matrix)

    def save_img_matrix(self):
        df = pd.DataFrame(self.conf_matrix, index=[c for c in "ABCDEJK"],
                          columns=[c for c in "ABCDEJK"])
        sn.heatmap(df, annot=True)

        plt.title("Matriz de confusao")
        plt.ylabel("Resposta real")
        plt.xlabel("Resposta obtida")

        plt.savefig("Confusion_matrix")
        plt.close()

    def plot_matrix(self, labels):
        self.conf_matrix /= self.conf_matrix.sum(axis=1)
        df = pd.DataFrame(self.conf_matrix, index=[c for c in labels],
                          columns=[c for c in labels])
        sn.heatmap(df, annot=True)

        plt.title("Matriz de confusao")
        plt.ylabel("Resposta real")
        plt.xlabel("Resposta obtida")

        plt.show()

    def load_matrix(self, matrix):
        self.conf_matrix = matrix
