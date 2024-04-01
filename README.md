## Half-life predictions in mammals utilizing BERT embeddings

The goal of this research was to see if the performance of Random Forest and XGBoost algorithms in predicting half-lives of mammals can be improved through the usage of 
BERT embeddings.

### Packages

Packages used for this project are included in the `requirements.txt` file. They can be downloaded by the following code:
```sh
   !pip install -r requirements.txt
   ```

### Data [^1]
- The datasets were taken from [this](https://github.com/vagarwal87/saluki_paper) research and can be found in this repository under the names, *all_HLs_human_featTable.txt* and 
 *all_HLs_mouse_featTable.txt*. 

 - ORF and Half-life values were extracted and saved in *Human_HL_ORF.csv* and *Mouse_HL_ORF.csv* by running the file `creating_tables.py`.
 - Each nucleotite in a sequence is separated by a whitespace to mimic the word and sentence relation with nucleotites and sequences. This is done for the BERT model.

 - One hot encoding of sequences is done in the file `one_hot_conversion.py` by assigning a vector to each nucleotite in the following way:
     - A : [1, 0, 0, 0]
     - G : [0, 1, 0, 0]
     - C : [0, 0, 1, 0]
     - T : [0, 0, 0, 1]

  The encoded sequences are saved as *Human_X.npy* and the corresponding Half-life values are saved as *Human_Y.npy*. 
  The file names in `one_hot_conversion.py` must be manually changed to create encodings for mouse data.

  - Embeddings are created from sequences by the BERT model in `embedding.py`. Embeddings are saved as *Human_EMBEDDINGSX.npy* and their corresponding
    Half-life values are saved as *Human_EMBEDDINGSY.npy*.

 [^1]: Due to Github's size limit, the .csv files in this repository are random samples. Therefore, the encodings and embeddings are also created based on the samples. 
 The project was done both on the original data and samples which gave similar results. 
 The original data can be found [here](https://drive.google.com/drive/folders/1Qc_cYvv983pQEqHL5Z6hvCjcMbFICTpx?usp=share_link).

### Algorithms

Random Forest and XGBoost are used to predict Half-lives with both encoded data and BERT embeddings as input. 
The data source must be manually changed in the files `random_forest.py` and `xgboost_model.py`. 

### Results

The performance of the algorithms were calculated based on R^2 and mean absolute error (MAE). There was no significant improvement in predictions.
