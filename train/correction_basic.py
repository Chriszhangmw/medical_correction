'''
这个版本得代码就直接用bert坐预测，但是这里得预训练模型，其实可以考虑换成一个针对医疗得预训练，比如用bert结合医疗
得语料做一些微调
'''

from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from train.config2 import Config
import numpy as np
import time
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class MaskedLM():
    def __init__(self,topK):
        self.topK = topK
        self.tokenizer = Tokenizer(Config.BERT_VOCAB_PATH,do_lower_case=True)
        self.model = build_transformer_model(Config.BERT_CONFIG_PATH,Config.BERT_CHECKPOINT_PATH,with_mlm = True)
        self.token_ids, self.segment_ids = self.tokenizer.encode(' ')

    def tokenizer_text(self,text):
        self.token_ids,self.segment_ids = self.tokenizer.encode(text)

    def find_topn_candidates(self,error_index):
        for i in error_index:
            self.token_ids[i] = self.tokenizer._token_dict['[MASK]'] #将待纠正的词用mask替换掉

        probs = self.model.predict([np.array([self.token_ids]),np.array([self.segment_ids])])[0]
        for i in range(len(error_index)):
            error_id = error_index[i]
            top_k_probs = np.argsort(-probs[error_id])[:self.topK]
            candidates,fin_prob = self.tokenizer.decode(top_k_probs),probs[error_id][top_k_probs]
            print(dict(zip(candidates,fin_prob)))

if __name__ == "__main__":
    masked_lm = MaskedLM(5)
    text = "刚刚一直再和老王谈天，他和他聊天很愉快"
    masked_lm.tokenizer_text(text)
    masked_lm.find_topn_candidates([9,12])





