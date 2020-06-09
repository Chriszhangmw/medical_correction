'''

这个版本的代码的思路还是用mask, 但是相当于用了阅读理解的方式，也用了微调，相比于上一个思路，就复杂些
在文本背后，添加一个和文本长度一样的mask串，然后预测这些mask，但是这个场景还是仅仅考虑错别词，并没有考虑多词少词的情况

'''

import json
from bert4keras.tokenizers import load_vocab, Tokenizer
from bert4keras.models import build_transformer_model
from bert4keras.snippets import DataGenerator, sequence_padding,AutoRegressiveDecoder
from bert4keras.optimizers import AdaFactor
from keras.layers import Lambda
from keras.models import Model
from keras.callbacks import Callback
import keras.backend as K
import numpy as np
from tqdm import tqdm
import random

input_reg = u', ., 。 ; " " ! ? [ ] 【 】 “ ” = @ # $ % & 0 1 2 3 4 5 6 7 8 9' \
            u''

out_reg = u''

#这里主要为了去除一些特殊字符





max_len = 64
batch_size = 8
epochs = 10
corpus_path = './data/train_all.txt'


config_path = '/home/chenbing/pretrain_models/bert/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = '/home/chenbing/pretrain_models/bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
vocab_path = '/home/chenbing/pretrain_models/bert/chinese_L-12_H-768_A-12/vocab.txt'

def format_text(input_text):
    '''
    标准化全角转半角，去除所有特殊符号
    :param input_text:
    :return:
    '''
    reg = {ord(f):ord(t) for f,t in zip(input_reg,output)}
    output_text = input_text.translate(reg)
    return output_text


def load_data(corpus_path):
    data = []
    with open(corpus_path,'r',encoding='utf-8') as rd:
        lines = rd.readlines()
        for line in lines:
            try:
                _,wrong,right = line.strip('\n').split('\t')
                wrong1 = format_text(wrong)[:max_len]
                right1 = format_text(right)[:max_len]

                if wrong1 == right1:
                    continue
                data.append((wrong1,right1))
            except Exception as err:
                print(line)
    return data
all_data = load_data(corpus_path)
random.shuffle(all_data)

valid_data = all_data[:len(all_data) // 8]
train_data = all_data[len(all_data) // 8:]




# 加载精简词表
token_dict, keep_words = load_vocab(
    dict_path=vocab_path,
    simplified=True,
    startswith=['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
)

tokenizer = Tokenizer(token_dict, do_lower_case=True)


class MyDataGenerator(DataGenerator):
    def __iter__(self, random=True):
        """
        单条样本格式: [cls]错误词汇[sep][mask][mask]..[sep]
        :param random:
        :return:
        """
        batch_tokens_ids, batch_segment_ids, batch_right_token_ids = [], [], []
        for is_end, D in self.sample(random):
            wrong, right = D
            right_token_ids, _ = tokenizer.encode(first_text=right)
            wrong_token_ids, _ = tokenizer.encode(first_text=wrong)

            token_ids = wrong_token_ids
            token_ids += [tokenizer._token_mask_id] * max_len
            token_ids += [tokenizer._token_end_id]

            segemnt_ids = [0] * len(token_ids)

            batch_tokens_ids.append(token_ids)
            batch_segment_ids.append(segemnt_ids)
            batch_right_token_ids.append(right_token_ids[1:])

            if len(batch_tokens_ids) == self.batch_size or is_end:
                batch_tokens_ids = sequence_padding(batch_tokens_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_right_token_ids = sequence_padding(batch_right_token_ids, max_len)

                yield [batch_tokens_ids, batch_segment_ids], batch_right_token_ids
                batch_tokens_ids, batch_segment_ids, batch_right_token_ids = [], [], []


# 构建模型
bert_model = build_transformer_model(
    config_path=config_path,
    checkpoint_path=checkpoint_path,
    with_mlm=True,
    keep_tokens=keep_words
)

output = Lambda(lambda x: x[:, 1:max_len + 1])(bert_model.output)
model = Model(bert_model.input, output)


def masked_cross_entropy(y_true, y_pred):
    """交叉熵作为loss，并mask掉padding部分的预测
    """
    y_true = K.reshape(y_true, [K.shape(y_true)[0], -1])
    y_mask = K.cast(K.not_equal(y_true, 0), K.floatx())
    cross_entropy = K.sparse_categorical_crossentropy(y_true, y_pred)
    cross_entropy = K.sum(cross_entropy * y_mask) / K.sum(y_mask)
    return cross_entropy


model.compile(loss=masked_cross_entropy, optimizer=AdaFactor(learning_rate=1e-3))
model.summary()


def ge_answer(wrong):
    """
    解码
    :param wrong:
    :return:
    """
    wrong_token_ids, _ = tokenizer.encode(wrong)
    token_ids = wrong_token_ids + [tokenizer._token_mask_id] * max_len + [tokenizer._token_end_id]
    segemnt_ids = [0] * len(token_ids)
    probas = model.predict([np.array([token_ids]), np.array([segemnt_ids])])[0][:len(wrong)]

    result = tokenizer.decode(probas.argmax(axis=1))
    return result




class Evaluator(Callback):
    def __init__(self):
        self.lowest = 1e10

    def on_epoch_end(self, epoch, logs=None):
        if logs['loss'] <= self.lowest:
            self.lowest = logs['loss']
            model.save('models/best_mlm_model.h5')







if __name__ == '__main__':
    #训练模型
    evaluator = Evaluator()
    train_generator = MyDataGenerator(train_data, batch_size=8)

    model.fit_generator(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=10,
        callbacks=[evaluator]
    )

    # predict
    model.load_weights('models/best_mlm_model.h5')
    wrong = '追风少俊年王俊凯'
    result = ge_answer(wrong)
    print(result)
