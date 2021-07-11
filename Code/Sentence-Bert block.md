# Sentence-BERT block



```python
class SBERT_Block(nn.Module):

    def __init__(self):
        super().__init__()

        self.bert_model = BertModel.from_pretraind('model')
        self.tokenizer = BertTokenizer.from_pretrained('model')
        self.max_len = max_len
        self.fc = nn.Linear(self.bert_model.config.hidden_size, fc_dim)
        self.bn = nn.BatchNorm1d(fc_dim)
        self._init_params()

    def _init_params(self):
        nn.init.xavier_normal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

    def extract_feat(self, x):
        tokenizer_output = self.tokenizer(x, truncation=True, padding=True, max_length=self.max_len)
 
        input_ids = torch.LongTensor(tokenizer_output['input_ids']).to('cuda')
        token_type_ids = torch.LongTensor(tokenizer_output['token_type_ids']).to('cuda')
        attention_mask = torch.LongTensor(tokenizer_output['attention_mask']).to('cuda')
        x = self.bert_model(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        x = torch.sum(x.last_hidden_state * attention_mask.unsqueeze(-1), dim=1) / attention_mask.sum(dim=1, keepdims=True)
        x = self.fc(x)
        x = self.bn(x)
        return x
```