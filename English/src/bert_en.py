from pytorch_transformers import (
    BertPreTrainedModel,
    BertModel,
)
from torch import nn, mean


class BertPooler(nn.Module):
    """This class is for pooling the model by taking the hidden state
    corresponding to the first token.
    """

    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertForSequenceMeanVec(BertPreTrainedModel):
    """This class is for creating the model from a pretrained bert configuration."""

    def __init__(self, config, num_labels, industry_num, state_dict=None):
        super(BertForSequenceMeanVec, self).__init__(config, state_dict=state_dict)
        self.num_labels = num_labels
        self.industry_num = industry_num

        self.bert = BertModel.from_pretrained(
            "bert-base-uncased", state_dict=state_dict
        )
        self.pooler_dev = BertPooler(config)
        self.pooler_stock = BertPooler(config)

        self.att = nn.Linear(self.bert.config.hidden_size, 1)
        self.att_industry = nn.Linear(self.bert.config.hidden_size, 1)
        self.dropout = nn.Dropout(self.bert.config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.bert.config.hidden_size, self.num_labels)
        self.classifier_industry = nn.Linear(
            self.bert.config.hidden_size, self.industry_num
        )
        self.bert.config.output_hidden_states = True
        self.bert.encoder.output_hidden_states = True

    def forward(
        self,
        input_array_doc,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        labels=None,
        label_industry=None,
    ):

        outputs = self.bert(
            input_array_doc,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )

        pooled_output = mean(outputs[0], axis=1)

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits, pooled_output)

        if type(label_industry) != type(None):
            logits_industry = self.classifier_industry(pooled_output)
            outputs += (logits_industry,)

        return outputs
