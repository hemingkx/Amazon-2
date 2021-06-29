from transformers.modeling_bert import *


class BertForCls(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForCls, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, inputs_embeds=None, head_mask=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)
        # (batch_size, sequence_length, hidden_size)
        sequence_output = outputs[0]
        # (batch_size, hidden_size)
        cls_output = sequence_output[:, 0, :]
        # dropout pred_label的一部分feature
        dropout_cls_output = self.dropout(cls_output)
        # 得到判别值, (batch_size, num_labels)
        logits = self.classifier(dropout_cls_output)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits, labels)
        outputs = (loss, logits)
        # contain: (loss), scores
        return outputs
