from pytorch_transformers import WEIGHTS_NAME, CONFIG_NAME
from pytorch_transformers.modeling_bert import BertForPreTraining, BertForSequenceClassification
from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule
from pytorch_transformers import BertPreTrainedModel, BertModel, BertForSequenceClassification, BertConfig
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class BertForSequenceMultiTaskAttention(BertPreTrainedModel):
    def __init__(self, config, num_labels, industry_num, state_dict = None):
        super(BertForSequenceMultiTaskAttention, self).__init__(config, state_dict = state_dict)
        # self.num_labels = config.num_labels
        self.num_labels = num_labels
        #self.num_labels_stock = num_labels_stock
        self.industry_num = industry_num

        self.bert = BertModel.from_pretrained('bert-base-uncased', state_dict = state_dict)
        self.pooler_dev = BertPooler(config)
        self.pooler_stock = BertPooler(config)
        
        self.att = nn.Linear(self.bert.config.hidden_size, 1)
        self.att_industry = nn.Linear(self.bert.config.hidden_size, 1)
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size,  self.num_labels )
        self.classifier_industry = nn.Linear(self.bert.config.hidden_size,  self.industry_num,)
        self.bert.config.output_hidden_states = True
        self.bert.encoder.output_hidden_states = True

    def forward(self, input_array_doc, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None, label_industry = None, labels_stock = None, barch_size = 32, stock_vol = 1):
        
        outputs = self.bert(input_array_doc,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids, 
                            head_mask=head_mask)
        
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        outputs = (logits, pooled_output)
        
        if type( label_industry) != type(None):
            logits_industry = self.classifier_industry(pooled_output)
            outputs += (logits_industry,)
        
        if type(labels_stock) != type(None):
            outputs += (logits_stock,)
        
        return outputs 

class BertForSequenceMeanVec(BertPreTrainedModel):
    def __init__(self, config, num_labels, industry_num, state_dict = None):
        super(BertForSequenceMeanVec, self).__init__(config, state_dict = state_dict)
        # self.num_labels = config.num_labels
        self.num_labels = num_labels
        #self.num_labels_stock = num_labels_stock
        self.industry_num = industry_num

        self.bert = BertModel.from_pretrained('bert-base-uncased', state_dict = state_dict)
        self.pooler_dev = BertPooler(config)
        self.pooler_stock = BertPooler(config)
        
        self.att = nn.Linear(self.bert.config.hidden_size, 1)
        self.att_industry = nn.Linear(self.bert.config.hidden_size, 1)
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size,  self.num_labels )
        self.classifier_industry = nn.Linear(self.bert.config.hidden_size,  self.industry_num,)
        self.bert.config.output_hidden_states = True
        self.bert.encoder.output_hidden_states = True

    def forward(self, input_array_doc, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None, label_industry = None, labels_stock = None, barch_size = 32, stock_vol = 1):
        
        outputs = self.bert(input_array_doc,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids, 
                            head_mask=head_mask)
        
        pooled_output = torch.mean(outputs[0], axis = 1)
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        outputs = (logits, pooled_output)
        
        if type( label_industry) != type(None):
            logits_industry = self.classifier_industry(pooled_output)
            outputs += (logits_industry,)
        
        if type(labels_stock) != type(None):
            outputs += (logits_stock,)
        
        return outputs 

