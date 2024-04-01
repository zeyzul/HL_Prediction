import pandas as pd

headers = ["ENSID", "HALFLIFE", "ORF"]

# LOAD DATA
df_mouse = pd.read_csv("all_HLs_mouse_featTable.txt", sep="\t")
df_human = pd.read_csv("all_HLs_human_featTable.txt", sep="\t")

# SAVE ONLY THE FIRST 100 NUCLEOTIDES AND REMOVE DUPLICATES
df_mouse["ORF"] = df_mouse["ORF"].str[:100]
df_mouse = df_mouse.drop_duplicates(subset=['ORF'])

df_human["ORF"] = df_human["ORF"].str[:100]
df_human = df_human.drop_duplicates(subset=['ORF'])

df_mouse = df_mouse[headers]
df_human = df_human[headers]

for i, row in df_mouse.iterrows():
    df_mouse.at[i,'ORF'] = " ".join(row["ORF"])

for i, row in df_human.iterrows():
    df_human.at[i,'ORF'] = " ".join(row["ORF"])

# REMOVE ONE ROW OF DIFFERENT LENGTH
df_human = df_human.drop(12587).reset_index()

# FOR FASTER RESULTS, A SAMPLE FROM DATA CAN ALSO BE USED
# df_human = df_human.sample(n=X)
# df_mouse = df_mouse.sample(n=X)

df_mouse.to_csv('Mouse_HL_ORF.csv', index = None, columns=headers)
df_human.to_csv('Human_HL_ORF.csv', index = None, columns=headers)












