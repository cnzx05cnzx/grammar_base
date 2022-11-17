import argparse
import random

from tqdm import tqdm
from tokenizers import BertWordPieceTokenizer
from pypinyin import pinyin, Style
import os
import sys

respo_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
if respo_path not in sys.path:
    sys.path.insert(0, respo_path)
import json
from allennlp.data.tokenizers import Token


class Pinyin(object):
    """docstring for Pinyin"""

    def __init__(self):
        super(Pinyin, self).__init__()
        self.shengmu = ['zh', 'ch', 'sh', 'b', 'p', 'm', 'f', 'd', 't', 'n', 'l', 'g', 'k', 'h', 'j', 'q', 'x', 'r',
                        'z', 'c', 's', 'y', 'w']
        self.yunmu = ['a', 'ai', 'an', 'ang', 'ao', 'e', 'ei', 'en', 'eng', 'er', 'i', 'ia', 'ian', 'iang', 'iao', 'ie',
                      'in', 'ing', 'iong', 'iu', 'o', 'ong', 'ou', 'u', 'ua', 'uai', 'uan', 'uang', 'ue', 'ui', 'un',
                      'uo', 'v', 've']
        self.shengdiao = ['1', '2', '3', '4', '5']
        self.sm_size = len(self.shengmu) + 1
        self.ym_size = len(self.yunmu) + 1
        self.sd_size = len(self.shengdiao) + 1

    def get_sm_ym_sd(self, pinyin):
        s = pinyin
        assert isinstance(s, str), 'input of function get_sm_ym_sd is not string'
        assert s[-1] in '12345', f'input of function get_sm_ym_sd is not valid,{s}'
        sm, ym, sd = None, None, None
        for c in self.shengmu:
            if s.startswith(c):
                sm = c
                break
        if sm == None:
            ym = s[:-1]
        else:
            ym = s[len(sm):-1]
        sd = s[-1]
        return sm, ym, sd

    def get_sm_ym_sd_labels(self, pinyin):
        sm, ym, sd = self.get_sm_ym_sd(pinyin)
        return self.shengmu.index(sm) + 1 if sm in self.shengmu else 0, \
               self.yunmu.index(ym) + 1 if ym in self.yunmu else 0, \
               self.shengdiao.index(sd) + 1 if sd in self.shengdiao else 0

    def get_pinyinstr(self, sm_ym_sd_label):
        sm, ym, sd = sm_ym_sd_label
        sm -= 1
        ym -= 1
        sd -= 1
        sm = self.shengmu[sm] if sm >= 0 else ''
        ym = self.yunmu[ym] if ym >= 0 else ''
        sd = self.shengdiao[sd] if sd >= 0 else ''
        return sm + ym + sd


class hanzi2pinyin():

    def __init__(self, chinese_bert_path, max_length: int = 512):
        self.vocab_file = os.path.join(chinese_bert_path, 'vocab.txt')
        self.config_path = os.path.join(chinese_bert_path, 'config')

        self.max_length = max_length
        self.tokenizer = BertWordPieceTokenizer(self.vocab_file)
        self.pho_convertor = Pinyin()
        # load pinyin map dict
        with open(os.path.join(self.config_path, 'pinyin_map.json'), encoding='utf8') as fin:
            self.pinyin_dict = json.load(fin)
        # load char id map tensor
        with open(os.path.join(self.config_path, 'id2pinyin.json'), encoding='utf8') as fin:
            self.id2pinyin = json.load(fin)
        # load pinyin map tensor
        with open(os.path.join(self.config_path, 'pinyin2tensor.json'), encoding='utf8') as fin:
            self.pinyin2tensor = json.load(fin)

    def convert_sentence_to_pinyin(self, sentence: str):
        # get pinyin of a sentence
        pinyin_list = pinyin(sentence, style=Style.TONE3, heteronym=True, errors=lambda x: [['not chinese'] for _ in x])
        # return [p[0] for p in pinyin_list]
        return [Token(p[0]) for p in pinyin_list]

    def convert_sentence_to_pinyin_ids(self, sentence: str, tokenizer_output):
        # get pinyin of a sentence
        pinyin_list = pinyin(sentence, style=Style.TONE3, heteronym=True, errors=lambda x: [['not chinese'] for _ in x])
        print(pinyin_list)
        pinyin_locs = {}
        # get pinyin of each location
        for index, item in enumerate(pinyin_list):
            pinyin_string = item[0]
            # not a Chinese character, pass
            if pinyin_string == "not chinese":
                continue
            if pinyin_string in self.pinyin2tensor:
                pinyin_locs[index] = self.pinyin2tensor[pinyin_string]
            else:
                ids = [0] * 8
                for i, p in enumerate(pinyin_string):
                    if p not in self.pinyin_dict["char2idx"]:
                        ids = [0] * 8
                        break
                    ids[i] = self.pinyin_dict["char2idx"][p]
                pinyin_locs[index] = ids

        # find chinese character location, and generate pinyin ids
        pinyin_ids = []
        for idx, (token, offset) in enumerate(zip(tokenizer_output.tokens, tokenizer_output.offsets)):
            if offset[1] - offset[0] != 1:
                pinyin_ids.append([0] * 8)
                continue
            if offset[0] in pinyin_locs:
                pinyin_ids.append(pinyin_locs[offset[0]])
            else:
                pinyin_ids.append([0] * 8)

        return pinyin_ids[:-1]

    def convert_sentence_to_shengmu_yunmu_shengdiao_ids(self, sentence, tokenizer_output):
        # get pinyin of a sentence
        pinyin_list = pinyin(sentence, style=Style.TONE3, neutral_tone_with_five=True,
                             heteronym=True, errors=lambda x: [['not chinese'] for _ in x])

        pinyin_locs = {}
        # get pinyin of each location
        for index, item in enumerate(pinyin_list):
            pinyin_string = item[0]
            # not a Chinese character, pass
            if pinyin_string == "not chinese":
                continue
            pinyin_locs[index] = self.pho_convertor.get_sm_ym_sd_labels(
                pinyin_string)

        # find chinese character location, and generate pinyin ids
        pinyin_labels = []
        # print(pinyin_locs)
        for idx, (token, offset) in enumerate(zip(tokenizer_output.tokens, tokenizer_output.offsets)):
            if offset[1] - offset[0] != 1:
                pinyin_labels.append((0, 0, 0))
                continue
            if offset[0] in pinyin_locs:
                pinyin_labels.append(pinyin_locs[offset[0]])
            else:
                pinyin_labels.append((0, 0, 0))

        return pinyin_labels

    def convert_sentence_to_s_y_s_ids(self, sentence):
        # get pinyin of a sentence
        pinyin_list = pinyin(sentence, style=Style.TONE3, neutral_tone_with_five=True,
                             heteronym=True, errors=lambda x: [['not chinese'] for _ in x])

        pinyin_labels = []
        # get pinyin of each location
        for index, item in enumerate(pinyin_list):
            pinyin_string = item[0]
            # not a Chinese character, pass
            if pinyin_string == "not chinese":
                pinyin_labels.append([0, 0, 0])
            else:
                pinyin_labels.append(list(self.pho_convertor.get_sm_ym_sd_labels(pinyin_string)))

        return pinyin_labels

    def convert_shengmu_yunmu_shengdiao_ids_to_pinyin_ids(self, sm_ym_sd_labels):

        pinyin_ids = []

        for sm_ym_sd_label in sm_ym_sd_labels:
            pinyin_str = self.pho_convertor.get_pinyinstr(sm_ym_sd_label)
            if pinyin_str == '':
                pinyin_ids.append([0] * 8)
            else:
                ids = [0] * 8
                for i, p in enumerate(pinyin_str):
                    if p not in self.pinyin_dict["char2idx"]:
                        ids = [0] * 8
                        break
                    ids[i] = self.pinyin_dict["char2idx"][p]
                pinyin_ids.append(ids)

        return pinyin_ids


if __name__ == '__main__':
    tokenizer = BertWordPieceTokenizer('../data/vocab.txt', lowercase=True)
    token2pinyin = hanzi2pinyin('../data')
    t = '我爱b北京天安门'
    encoded = tokenizer.encode(t)
    # print(token2pinyin.get_sm_ym_sd_labels(''))
    print(token2pinyin.convert_sentence_to_pinyin_ids(t, encoded))
    print(token2pinyin.convert_sentence_to_s_y_s_ids(t))
    print(token2pinyin.convert_sentence_to_shengmu_yunmu_shengdiao_ids(t, encoded))
