import numpy as np
from sklearn.metrics import classification_report, pairwise
from torch import LongTensor


def get_input_ids_from_text(input_text, tokenizer):
    """This is a function to get ids from input text"""
    tokenized_text = tokenizer.tokenize(input_text)
    tokenized_id = tokenizer.convert_tokens_to_ids(tokenized_text)
    return tokenized_id


def get_input_text_from_ids(input_ids, tokenizer):
    """This is a function to get text from ids"""
    tokenized_text = tokenizer.convert_ids_to_tokens(input_ids)
    return tokenized_text


def evaluate_model_sector_prediction(
    model,
    test_data_x,
    test_data_y,
    test_data_industry,
    test_data_size,
    mode_classifier=True,
    max_seq_length=512,
    batch_size=8,
):
    """This is a function to predict the sector given the input text ids"""
    model = model.eval()

    pred_label_test = []
    answer_label_test = []
    pred_industry_test = []
    answer_indesutry_test = []
    pred_label_prob_list = []
    pred_industry_prob_list = []

    for data_index in range(0, len(test_data_x), batch_size):
        data_batch = test_data_x[data_index : data_index + batch_size]
        doc_batch = [doc[0] for doc in data_batch]
        logits = 0
        industry_logits_all = 0

        """formatting the input data"""
        input_array_doc = []
        for doc_batch_index, input_ids in enumerate(doc_batch):
            input_array = np.zeros(max_seq_length, dtype=np.int)
            input_array[: min(max_seq_length, 1)] = input_ids[: min(max_seq_length, 1)]
            input_array_doc.append(input_array)
        input_ids = LongTensor(np.array(input_array_doc).astype(np.int32))

        """getting the model's output"""
        label_logits, industry_logits = model(input_ids)

        """getting the values of the predicted probabilities"""
        logits += label_logits
        industry_logits_all += industry_logits
        pred_label = np.argmax(logits.detach().to("cpu").numpy(), axis=1)
        pred_industry = np.argmax(
            industry_logits_all.detach().to("cpu").numpy(), axis=1
        )

        """creating the output lists for the predicted values"""
        pred_label_test += list(pred_label)
        pred_industry_test += list(pred_industry)
        answer_label_test += list(test_data_y[data_index : data_index + batch_size])
        answer_indesutry_test += list(
            test_data_industry[data_index : data_index + batch_size]
        )

    """printing classification metrics of the sectors"""
    target_sectors = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    print(classification_report(answer_label_test, pred_label_test, target_sectors))

    return (
        pred_label_test,
        answer_label_test,
        pred_industry_test,
        answer_indesutry_test,
    )


def evaluate_model_sector_similarity(
    model,
    test_data_x,
    test_data_y,
    test_data_size,
    mode_classifier=True,
    batch_size=16,
    max_seq_length=512,
):
    """This is a function to predict the sector similarity given the input text ids"""
    model = model.eval()

    """formatting the input data"""
    sentence_representation_all = []
    for data_index in range(0, len(test_data_x), batch_size):
        data_batch = test_data_x[data_index : data_index + batch_size]
        doc_batch = [doc[0] for doc in data_batch]
        logits = 0
        industry_logits_all = 0
        label_batch = np.array(test_data_y[data_index : data_index + batch_size])
        input_array_doc = []
        for doc_batch_index, input_ids in enumerate(doc_batch):
            input_array = np.zeros(max_seq_length, dtype=np.int)
            input_array[: min(max_seq_length, len(input_ids))] = input_ids[
                : min(max_seq_length, len(input_ids))
            ]
            input_array_doc.append(input_array)
        input_ids = LongTensor(np.array(input_array_doc).astype(np.int32))

        """getting the model's output"""
        label_logits, pooled_output = model(input_ids, labels=LongTensor(label_batch))
        sentence_representation_all.append(pooled_output.detach().to("cpu").numpy())

    """getting the values of the predicted probabilities"""
    sentence_representation_all = np.vstack(sentence_representation_all)

    """creating a similarity matrix for the predicted labels"""
    doc2soc_similarity_mat = pairwise.cosine_similarity(sentence_representation_all)

    """setting the similarity between the same company as 0"""
    for i in range(len(doc2soc_similarity_mat)):
        doc2soc_similarity_mat[i, i] = 0

    """creating a similarity matrix for the actual labels.
    The matrix contains a bool to check if 2 companies belong to the same sector"""
    same_or_other_label_mat = (
        np.array(test_data_y)[:, np.newaxis] == np.array(test_data_y)[np.newaxis, :]
    )

    """printing the probability of 2 companies belonging to the same/other sector"""
    print(
        "same: ",
        (
            doc2soc_similarity_mat[same_or_other_label_mat][
                doc2soc_similarity_mat[same_or_other_label_mat] != 0
            ]
        ).mean(),
    )
    print(
        "other: ",
        (
            doc2soc_similarity_mat[same_or_other_label_mat == False][
                doc2soc_similarity_mat[same_or_other_label_mat == False] != 0
            ]
        ).mean(),
    )
    return doc2soc_similarity_mat
