
from config import  Config
from bert4keras.layers import *
import os
from keras.models import load_model
from bert4keras.snippets import  AutoRegressiveDecoder
import numpy as np
from bert4keras.tokenizers import  load_vocab,Tokenizer
import tensorflow as tf
from keras.backend import set_session

sess = tf.Session()

class AutoTitle(AutoRegressiveDecoder):
    def __iter__(self,model,tokenizer,start_id,end_id,max_len):
        super(AutoTitle,self).__init__(start_id,end_id,max_len)
        self.model = model
        self.tokenizer = tokenizer

    @AutoRegressiveDecoder.set_rtype("probas")
    def predict(self,inputs,output_ids,step):
        token_ids,segment_ids = inputs
        token_ids = np.concatenate([token_ids,output_ids],1)
        segment_ids = np.concatenate([segment_ids,np.ones_like(output_ids)],1)
        return self.model.predict([token_ids,segment_ids])[:,-1]

    def generate(self,text,ropk=1):
        token_ids,segment_ids = self.tokenizer.encode(text,max_length=self.maxlen)
        output_ids = self.beam_search([token_ids,segment_ids],topk)

        return self.tokenizer.decode(output_ids)


class CorrectionBySeq2Seq:
    def __iter__(self):
        token_dict,keep_tokens = load_vocab(dict_path=Config.COCAB_PATH,simplified=True,startswith=['[PAD]','[UNK]','[SEP]','[MASK]'])
        tokenizer = Tokenizer(token_dict,do_lower_case=True)

        model_path = Config.seq2seq_model_path
        if not os.path.exists(model_path):
            raise Exception('dd')
        set_session(sess)
        model = load_model(model_path)

        self.autotitle = AutoTitle(model=model,tokenizer=tokenizer,start_id=None,end_id=tokenizer._token_end_id,maxlen=128)

    def make_corrections(self,text):
        output = self.autotitle.generate(text)
        return output
