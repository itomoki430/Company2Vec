import sys

sys.path.append("./src")

import ml_metrics
import numpy as np
import os
import pandas as pd
from pytorch_transformers import BertConfig
import sklearn
from torch import load, no_grad

from bert_en import BertForSequenceMeanVec
from evaluation import evaluate_model_sector_similarity

"""Path to the fine-tuned model on all sectors"""
abs_model_path = "/home/jihene/Accelex_repo/Company2Vec/English/models"
read_file_name = os.path.join(
    abs_model_path,
    "20200120_BERTMeanVEC/Stock_Sector_SectorName_1/01/pytorch_model.bin",
)

max_seq_length = 512
batch_size = 1

"""Path to the data files: 
df_ticker contains: ticker / sector / industry / text description
text_data_id_df contains: ticker / text description ids """

abs_data_path = "/home/jihene/Accelex_repo/Company2Vec/English/data"
df_ticker = pd.read_csv(os.path.join(abs_data_path, "ticker_list.csv"))
text_data_id_df = pd.read_csv(os.path.join(abs_data_path, "text_data_id.csv"))

"""Creating dictionaries for the ticker/sector, ticker/industry and ticker/text_id"""
ticker2sector = dict(zip(df_ticker["ticker"], df_ticker["sector"]))
ticker2industry = dict(zip(df_ticker["ticker"], df_ticker["industry"]))
text_data = dict(
    zip(
        text_data_id_df["ticker"],
        [
            [
                np.array(text_data_id_df["text_id"].iloc[i][1:-1].split(",")).astype(
                    np.int32
                )
            ]
            for i in range(len(text_data_id_df))
        ],
    )
)

"""Creating lists for the sector, the industry and the text_id"""
ticker_list = ticker2sector.keys()
sector_array = []
industry_array = []
use_text_data = []

for index, ticker in enumerate(ticker_list):
    sector_array.append(ticker2sector[ticker])
    use_text_data.append(text_data[ticker])
    industry_array.append(ticker2industry[ticker])

"""Creating dictionaries for sorted sector/id and industry/id """
sector2id = dict(zip(np.sort(list(set(sector_array))), range(len(set(sector_array)))))
industry2id = dict(
    zip(np.sort(list(set(industry_array))), range(len(set(industry_array))))
)

"""Creating lists for:
use_label_data: contains the sector labels
use_industry_data: contains the industry labels"""
use_label_data = [sector2id[sector] for sector in sector_array]
use_industry_data = [industry2id[sector] for sector in industry_array]

"""Specifying data size"""
test_data_size = 3  # len(ticker_list)

"""Specifying the input data for the model"""
test_data_x = use_text_data[-test_data_size:]
test_data_y = use_label_data[-test_data_size:]
test_data_industry = use_industry_data[-test_data_size:]
label_num = max(use_label_data) + 1
industry_num = max(use_industry_data) + 1

"""loading the model's checkpoints"""
bert_config = BertConfig.from_pretrained(
    os.path.join(abs_model_path, "PreTrainigNLMbyBert/finetuned_lm/")
)
model_state_dict = load(read_file_name, map_location="cpu")
model = BertForSequenceMeanVec(
    bert_config, label_num, industry_num, state_dict=model_state_dict
)

"""making the predictions on the input test data"""
with no_grad():
    sentence_representation_all_test = evaluate_model_sector_similarity(
        model, test_data_x, test_data_y, test_data_size
    )

    """getting the similarity matrix between the input text_ids considering the sector
    """
    doc2soc_similarity_mat = sklearn.metrics.pairwise.cosine_similarity(
        sentence_representation_all_test
    )
    print("similarity matrix:")
    print(doc2soc_similarity_mat)

    """sorting the similarity matrix"""
    sort_values = np.argsort(doc2soc_similarity_mat, axis=1)[:, -1::-1]

    """getting the predicted values and the actual values"""
    print("sector (MAP@K)")
    predicted_list = [list(item[1:]) for item in sort_values]
    actual_list_all = [
        list(
            np.array(range(len(test_data_y)))[
                np.array(test_data_y) == test_data_y[index]
            ]
        )
        for index in range(len(test_data_y))
    ]
    actual_list_rev = []
    for index, item in enumerate(actual_list_all):
        actual_list_rev.append(list(np.array(item)[np.array(item) != index]))

    """computing the mean average precision at k for the predictions.
    k is the number of elements of "predicted" to consider in the calculation
    which is the top_n companies extracted from the sorted_values
    """
    for top_n in [5, 10, 50]:
        print(top_n, ":", ml_metrics.mapk(actual_list_rev, predicted_list, top_n))
