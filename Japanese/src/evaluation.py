import collections
import logging
import os
import torch
import MeCab
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, WordpieceTokenizer
from pytorch_pretrained_bert.tokenization import load_vocab
from pytorch_transformers import BertPreTrainedModel, BertModel, BertForSequenceClassification, BertConfig
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import pandas as pd
import numpy as np
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule
from collections import defaultdict
import sklearn
import sklearn.metrics
import  matplotlib.pyplot as plt
import time
from sklearn.metrics import classification_report
import copy
from multiprocessing import Pool, cpu_count
import requests
import ml_metrics 
import pickle
import sys
import datetime
from dateutil.relativedelta import relativedelta
from torch import functional as F

loss_fct = CrossEntropyLoss(ignore_index = -1)
loss_mse = MSELoss()

def evaluate_model(model, test_data_x, test_data_y, test_data_industry, test_data_size, mode_classifier = True, batch_size = 8, max_seq_length = 512):
    model = model.eval()
    tr_loss_test = 0
    #score_all = []
    pred_label_test = []
    answer_label_test = []
    pred_industry_test = []
    answer_indesutry_test = []
    for data_index in range(0, len(test_data_x), batch_size):
        data_batch = test_data_x[data_index:data_index+batch_size]
        doc_batch = [doc for doc in data_batch]
        logits = 0
        industry_logits_all = 0
        label_batch = np.array(test_data_y[data_index:data_index + batch_size])
        input_array_doc = []
        for doc_batch_index, input_ids in enumerate(doc_batch):
                input_array = np.zeros(max_seq_length, dtype=np.int)
                input_array[:min(max_seq_length, len(input_ids))] = input_ids[:min(max_seq_length, len(input_ids))]
                input_array_doc.append(input_array)
            
        #print (len(input_array_doc))
        input_ids = torch.LongTensor(np.array(input_array_doc).astype(np.int32))
        label_logits, pooled_output, industry_logits = model(input_ids, labels= torch.LongTensor(label_batch),  
                                                label_industry = torch.LongTensor(test_data_industry),
                                                labels_stock =  None, stock_vol = 0)
        
        logits  += label_logits
        industry_logits_all += industry_logits
        
        loss = loss_fct(
            logits,  
            torch.LongTensor(np.array(test_data_y[data_index:data_index + batch_size])).to("cuda")).detach().to("cpu").item()   
        
        pred_label = np.argmax(logits.detach().to("cpu").numpy(), axis = 1)
        pred_industry = np.argmax(industry_logits_all.detach().to("cpu").numpy(), axis = 1)
        pred_label_test+= list(pred_label )
        pred_industry_test+= list(pred_industry)
        answer_label_test += list(test_data_y[data_index:data_index+batch_size])
        answer_indesutry_test+= list(test_data_industry[data_index:data_index+batch_size])
        tr_loss_test += loss
    return pred_label_test, answer_label_test, pred_industry_test, answer_indesutry_test, tr_loss_test

def evaluate_model_with_cosine_similarity(
    model, test_data_x, test_data_y, test_data_industry, test_data_size, mode_classifier = True, batch_size = 8, max_seq_length = 512):
    model = model.eval()
    tr_loss_test = 0
    #score_all = []
    pred_label_test = []
    answer_label_test = []
    pred_industry_test = []
    answer_indesutry_test = []
    sentence_representation_all = []
    for data_index in range(0, len(test_data_x), batch_size):
        data_batch = test_data_x[data_index:data_index+batch_size]
        doc_batch = [doc for doc in data_batch]
        logits = 0
        industry_logits_all = 0
        label_batch = np.array(test_data_y[data_index:data_index + batch_size])
        input_array_doc = []
        for doc_batch_index, input_ids in enumerate(doc_batch):
                input_array = np.zeros(max_seq_length, dtype=np.int)
                input_array[:min(max_seq_length, len(input_ids))] = input_ids[:min(max_seq_length, len(input_ids))]
                input_array_doc.append(input_array)
            
        #print (len(input_array_doc))
        input_ids = torch.LongTensor(np.array(input_array_doc).astype(np.int32))
        label_logits, pooled_output, industry_logits = model(input_ids, labels= torch.LongTensor(label_batch),  
                                                label_industry = torch.LongTensor(test_data_industry),
                                                labels_stock =  None, stock_vol = 0)
        
        
        #print (pooled_output.detach().to("cpu").numpy().shape)
        sentence_representation_all.append(pooled_output.detach().to("cpu").numpy())
    
    
    
    sentence_representation_all = np.vstack(sentence_representation_all)
    doc2soc_similarity_mat = sklearn.metrics.pairwise.cosine_similarity(sentence_representation_all)
    for i in range(len(doc2soc_similarity_mat)):
        doc2soc_similarity_mat[i,i] = 0 
        
    same_or_other_label_mat = (np.array(test_data_y)[:,np.newaxis] == np.array(test_data_y)[np.newaxis, :])
    print ("sector")
    print ("same: ", (doc2soc_similarity_mat[same_or_other_label_mat][doc2soc_similarity_mat[same_or_other_label_mat] !=0]).mean())
    print ("other: ", (doc2soc_similarity_mat[same_or_other_label_mat == False][
        doc2soc_similarity_mat[same_or_other_label_mat== False] !=0]).mean())
    
    same_or_other_industry_mat = (np.array(test_data_industry)[:,np.newaxis] == np.array(test_data_industry)[np.newaxis, :])
    print ("industry")
    print ("same: ", (doc2soc_similarity_mat[same_or_other_industry_mat][doc2soc_similarity_mat[same_or_other_industry_mat] != 0]).mean())
    print ("other: ", (doc2soc_similarity_mat[same_or_other_industry_mat == False][doc2soc_similarity_mat[same_or_other_industry_mat == False] !=0]).mean())
    
    
    return sentence_representation_all