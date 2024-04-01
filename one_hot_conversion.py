import numpy as np
import pandas as pd

# CHANGE THE FILE MANUALLY: 'Human_HL_ORF.csv' / 'Mouse_HL_ORF.csv'
data = pd.read_csv('Human_HL_ORF.csv')

ORF_values = data["ORF"].values


# REMOVE WHITESPACE BETWEEN NUCLEOTIDES
for element in ORF_values:
    new_element = element.replace(" ", "")
    ORF_values[ORF_values == element] = new_element


def conversion(row):
    nuc_vector = np.array([])

    for i in row:
        if i == "A":
            nuc_vector = np.concatenate((nuc_vector, [1, 0, 0, 0]))
        elif i == "G":
            nuc_vector = np.concatenate((nuc_vector, [0, 1, 0, 0]))
        elif i == "C":
            nuc_vector = np.concatenate((nuc_vector, [0, 0, 1, 0]))
        elif i == "T":
            nuc_vector = np.concatenate((nuc_vector, [0, 0, 0, 1]))

    return nuc_vector


converted = []
for element in ORF_values:
    converted_row = conversion(element)
    converted.append(converted_row)

converted = np.array(converted)
print(converted.shape)

HL_values = data["HALFLIFE"]
HL_values = HL_values.to_numpy()


# CHANGE FILE NAMES MANUALLY: "HUMAN_X.npy" - "HUMAN_Y.npy" / "MOUSE_X.npy" - "MOUSE_Y.npy"
file_name_x = "HUMAN_X.npy"
file_name_y = "HUMAN_Y.npy"

np.save(file_name_x, converted)
np.save(file_name_y, HL_values)

