import torch
from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# THE ORF VALUES
def create_seq_list(HL_df):
    seq_values = []

    for i in range(0, len(HL_df)):
        seq_values.append(HL_df["ORF"][i])

    print("Created list from sequences...")
    return seq_values


# HALFLIFE
def HL_values(HL_df, column):
    return HL_df[column].values.tolist()


# TOKENIZATION
def tokenize_seq(seq_list):
    tokenized = []

    for i in range(0, len(seq_list)):
        row_token = tokenizer.encode_plus(seq_list[i],
                                          max_length=102,
                                          add_special_tokens=True,  # Add '[CLS]' and '[SEP]'
                                          truncation=True,
                                          return_token_type_ids=False,
                                          padding='max_length',
                                          return_attention_mask=False,
                                          return_tensors='pt')

        row_token = tokenizer.convert_ids_to_tokens(row_token['input_ids'][0])
        tokenized.append(row_token)

    print("Tokenized sequences...")
    return tokenized


# SEGMENT IDS
def index_seq(tokenized_list):  # Assigns rows ids 1 and 0 consecutively.
    indexed = []

    for i in range(0, len(tokenized_list)):
        indexed.append(tokenizer.convert_tokens_to_ids(tokenized_list[i]))

    print("Indexed sequences...")
    return indexed


def segment_ids_seq(tokenized_list):
    segment_ids = []

    # First row is 0
    even = True

    for i in range(0, len(tokenized_list)):
        if even:
            segment_ids.append([1] * len(tokenized_list[i]))
        else:
            segment_ids.append([0] * len(tokenized_list[i]))

        even = not even

    print("Created segments IDs form sequences...")
    return segment_ids


# RUNNING BERT
def create_embeddings(indexed_list, segment_id_list):
    print("Running bert model...")
    tokens_tensor = torch.tensor([indexed_list])  # tokens_tensor.shape = (1, total_rows, 512)
    segments_tensors = torch.tensor([segment_id_list])

    tokens_tensor = torch.squeeze(tokens_tensor)
    segments_tensors = torch.squeeze(segments_tensors)

    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    model.eval()

    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        hidden_states = outputs[2]

    return hidden_states


# CREATING EMBEDDINGS
def create_vectors(model_output, sequences):
    print("Creating embeddings from the output...")
    datatset_embeddings = np.array([])

    for j in range(0, len(sequences)):
        a = np.array([])
        for i in range(0, 102):
            b = (np.array(model_output[12][j][i]) + np.array(model_output[11][j][i]) + np.array(
                model_output[10][j][i]) + np.array(model_output[9][j][i])) / 4
            a = np.hstack((a, b))
        if len(datatset_embeddings) == 0:
            datatset_embeddings = a
        else:
            datatset_embeddings = np.vstack((datatset_embeddings, a))

    datatset_embeddings = pd.DataFrame(datatset_embeddings)

    return datatset_embeddings


def final_embedding(HL_df):
    rows = create_seq_list(HL_df)
    tokenized = tokenize_seq(rows)
    indexed = index_seq(tokenized)
    segment_id = segment_ids_seq(tokenized)

    output = create_embeddings(indexed, segment_id)
    embedding = create_vectors(output, rows)

    return embedding


human = pd.read_csv("Human_HL_ORF.csv")
mouse = pd.read_csv("Mouse_HL_ORF.csv")


human_embedding = final_embedding(human)

file_name_x = "HUMAN_EMBEDDINGSX.npy"
file_name_y = "HUMAN_EMBEDDINGSY.npy"

human_HL_values = HL_values(human, "HALFLIFE")

np.save(file_name_x, human_embedding)
np.save(file_name_y, human_HL_values)


mouse_embedding = final_embedding(mouse)

file_name_x = "MOUSE_EMBEDDINGSX.npy"
file_name_y = "MOUSE_EMBEDDINGSY.npy"

mouse_HL_values = HL_values(mouse, "HALFLIFE")

np.save(file_name_x, mouse_embedding)
np.save(file_name_y, mouse_HL_values)

