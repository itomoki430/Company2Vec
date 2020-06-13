from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM, WordpieceTokenizer
from pytorch_pretrained_bert.tokenization import load_vocab
from pytorch_transformers import BertPreTrainedModel, BertModel, BertForSequenceClassification, BertConfig
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule
import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import os
import collections
import MeCab

DICDIR = "dictionary/mecab-ipadic-neologd/build/mecab-ipadic-2.7.0-20070801-neologd-20191111"
USERDIC = 'dictionary/user.dic, userdictionaryruiter-keyword.dic'

class MeCabBert:
    def __init__(self, dicdir=DICDIR, userdic=USERDIC):
        self.tagger = MeCab.Tagger(f"-Owakati -u {userdic}")
        #self.tagger = MeCab.Tagger(f"-d {dicdir} -b 100000 -Owakati -u {userdic}")
        #self.tagger = MeCab.Tagger(f"-d {dicdir} -b 100000 -Owakati -u user.dic")
        self.tagger.parse("")

    def tokenize(self, text):
        return self.tagger.parse(text).split()
        
class BertMeCabTokenizer(BertTokenizer):
    def __init__(self, vocab_file, do_lower_case=True, max_len=None, do_basic_tokenize=True,
                 never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
        """Constructs a BertTokenizer.
        Args:
          vocab_file: Path to a one-wordpiece-per-line vocabulary file
          do_lower_case: Whether to lower case the input
                         Only has an effect when do_wordpiece_only=False
          do_basic_tokenize: Whether to do basic tokenization before wordpiece.
          max_len: An artificial maximum length to truncate tokenized sequences to;
                         Effective maximum length is always the minimum of this
                         value (if specified) and the underlying BERT model's
                         sequence length.
          never_split: List of tokens which will never be split during tokenization.
                         Only has an effect when do_wordpiece_only=False
        """
        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`".format(vocab_file))
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])
        self.do_basic_tokenize = do_basic_tokenize
        if do_basic_tokenize:
            self.basic_tokenizer = MeCabBert()
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
        self.max_len = max_len if max_len is not None else int(1e12)

    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            ids.append(self.vocab.get(token, 1))
        if len(ids) > self.max_len:
            logger.warning(
                "Token indices sequence length is longer than the specified maximum "
                " sequence length for this BERT model ({} > {}). Running this"
                " sequence through BERT will result in indexing errors".format(len(ids), self.max_len)
            )
        return ids

class BertForSequenceMultiTaskAttention(BertPreTrainedModel):
    def __init__(self, config, num_labels, industry_num, state_dict = None):
        super(BertForSequenceMultiTaskAttention, self).__init__(config, read_f_name = './models/PyTorchPretrainendModel', state_dict = state_dict)
        # self.num_labels = config.num_labels
        self.num_labels = num_labels
        #self.num_labels_stock = num_labels_stock
        self.industry_num = industry_num

        self.bert = BertModel.from_pretrained('./models/PyTorchPretrainendModel', state_dict = state_dict)
        self.pooler_dev = BertPooler(config)
        self.pooler_stock = BertPooler(config)
        
        self.att = nn.Linear(self.bert.config.hidden_size, 1)
        self.att_industry = nn.Linear(self.bert.config.hidden_size, 1)
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size,  self.num_labels )
        self.classifier_industry = nn.Linear(self.bert.config.hidden_size,  self.industry_num,)
        

        #self.init_weights()
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
        #pooled_output_dev = self.pooler_dev(outputs[2][-1])
        #pooled_output_stock = self.pooler_stock(outputs[2][-1])
        # pooled_output = torch.mean(torch.stack([(self.bert.pooler(hidden)) for hidden in outputs[2][1:]]), axis = 0)
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        outputs = (logits, pooled_output)
        """
        if type(labels) != type(None):
            logits = self.metric_fc(pooled_output, labels)
            outputs = (logits, pooled_output)
        else:
            outputs = (None, pooled_output)
        """
        if type( label_industry) != type(None):
            #logits_industry = self.classifier_industry(pooled_output_dev)
            logits_industry = self.classifier_industry(pooled_output)
            outputs += (logits_industry,)
        
        if type(labels_stock) != type(None):
            #logits_stock = self.classifier_stock(pooled_output_stock)
            #logits_stock = self.classifier_stock(pooled_output)
            outputs += (logits_stock,)
        
        return outputs 

    
class BertForSequenceMeanVec(BertPreTrainedModel):
    def __init__(self, config, num_labels, industry_num, state_dict = None):
        super(BertForSequenceMeanVec, self).__init__(config, read_f_name = './models/PyTorchPretrainendModel', state_dict = state_dict)
        # self.num_labels = config.num_labels
        self.num_labels = num_labels
        #self.num_labels_stock = num_labels_stock
        self.industry_num = industry_num

        self.bert = BertModel.from_pretrained('./models/PyTorchPretrainendModel', state_dict = state_dict)
        self.pooler_dev = BertPooler(config)
        self.pooler_stock = BertPooler(config)
        
        self.att = nn.Linear(self.bert.config.hidden_size, 1)
        self.att_industry = nn.Linear(self.bert.config.hidden_size, 1)
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size,  self.num_labels )
        self.classifier_industry = nn.Linear(self.bert.config.hidden_size,  self.industry_num,)
        #self.classifier_stock = nn.Linear(self.bert.config.hidden_size,  self.num_labels_stock)
        

        #self.init_weights()
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
        #pooled_output = outputs[1]
        #pooled_output_dev = self.pooler_dev(outputs[2][-1])
        #pooled_output_stock = self.pooler_stock(outputs[2][-1])
        # pooled_output = torch.mean(torch.stack([(self.bert.pooler(hidden)) for hidden in outputs[2][1:]]), axis = 0)
        
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        outputs = (logits, pooled_output)
        """
        if type(labels) != type(None):
            logits = self.metric_fc(pooled_output, labels)
            outputs = (logits, pooled_output)
        else:
            outputs = (None, pooled_output)
        """
        if type( label_industry) != type(None):
            #logits_industry = self.classifier_industry(pooled_output_dev)
            logits_industry = self.classifier_industry(pooled_output)
            outputs += (logits_industry,)
        
        if type(labels_stock) != type(None):
            #logits_stock = self.classifier_stock(pooled_output_stock)
            #logits_stock = self.classifier_stock(pooled_output)
            outputs += (logits_stock,)
        
        return outputs 

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