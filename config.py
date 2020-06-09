import os
import json

#获取当前文件绝对目录的上级目录，即项目目录（config.py文件默认存放在项目目录下）
PROJECT_PATH = os.path.abspath(os.path.dirname(__file__))





class Config:

    is_develop = json.load(open(PROJECT_PATH + '/branch.json'))["is_develop"]

    if is_develop:
        '''
        发布环境
        '''
        ALBERT_CONFIG_PATH = "/home/ai/pre_models/albert_samll_zh_google/albert_config_samll_google.json"
        ALBERT_CHECKPOINT_PATH = "/home/ai/pre_models/albert_samll_zh_google/albert_model.ckpt"
        ALBERT_VOCAB_PATH = "/home/ai/pre_models/albert_samll_zh_google/vocab.txt"

        BERT_CONFIG_PATH = "/home/ai/pre_models/bert/chinese_L-12_H-768_A-12/bert_config.json"
        BERT_CHECKPOINT_PATH = "/home/ai/pre_models/bert/chinese_L-12_H-768_A-12/bert_model.ckpt"
        BERT_VOCAB_PATH = "/home/ai/pre_models/bert/chinese_L-12_H-768_A-12/vocab.txt"

        ELECTRA_CONFIG_PATH = "/home/ai/pre_models/electra/chinese_electra_small_L-12_H-256_A-12/electra_config.json"
        ELECTRA_CHECKPOINT_PATH = "/home/ai/pre_models/electra/chinese_electra_small_L-12_H-256_A-12/electra_samll"
        ELECTRA_VOCAB_PATH = "/home/ai/pre_models/electra/chinese_electra_small_L-12_H-256_A-12/vocab.txt"

    else:
        '''
        开发环境
        '''
        ALBERT_CONFIG_PATH = "/home/zhangmeiwei/pre_models/albert_samll_zh_google/albert_config_samll_google.json"
        ALBERT_CHECKPOINT_PATH = "/home/zhangmeiwei/pre_models/albert_samll_zh_google/albert_model.ckpt"
        ALBERT_VOCAB_PATH = "/home/zhangmeiwei/pre_models/albert_samll_zh_google/vocab.txt"

        BERT_CONFIG_PATH = "/home/zhangmeiwei/pre_models/bert/chinese_L-12_H-768_A-12/bert_config.json"
        BERT_CHECKPOINT_PATH = "/home/zhangmeiwei/pre_models/bert/chinese_L-12_H-768_A-12/bert_model.ckpt"
        BERT_VOCAB_PATH = "/home/ai/pre_models/bert/chinese_L-12_H-768_A-12/vocab.txt"

        ELECTRA_CONFIG_PATH = "/home/zhangmeiwei/pre_models/electra/chinese_electra_small_L-12_H-256_A-12/electra_config.json"
        ELECTRA_CHECKPOINT_PATH = "/home/zhangmeiwei/pre_models/electra/chinese_electra_small_L-12_H-256_A-12/electra_samll"
        ELECTRA_VOCAB_PATH = "/home/zhangmeiwei/pre_models/electra/chinese_electra_small_L-12_H-256_A-12/vocab.txt"

    seq2seq_model_path = os.path.join(PROJECT_PATH,"models/parallel_model_deq2seq.h5")

