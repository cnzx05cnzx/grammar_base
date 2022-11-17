import torch
import random
import numpy as np
from torch.nn import CrossEntropyLoss


def test_cude():
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())


def align():
    with open('./data/gec/test_add.txt', mode='r', encoding='utf-8') as f1, \
            open('./data/gec/test_new.txt', mode='w', encoding='utf-8', newline='') as f2:
        for f in f1:
            f = list(f.strip())
            f2.write(' '.join(f) + '\r\n')


def make_data():
    input_ids = [[101, 4664, 4719, 5381, 102], [101, 4664, 49, 102, 0]]
    input_ids = torch.LongTensor(input_ids)
    labels = [[101, 4664, 4719, 5320, 102], [101, 4664, 23, 102, 0]]
    labels = torch.LongTensor(labels)

    loss_fct = CrossEntropyLoss(reduction='none')
    loss_mask = (input_ids != 0) * (input_ids != 101) * (input_ids != 102).long()
    active_loss = loss_mask.view(-1) == 1
    print(loss_mask)
    print(active_loss)
    print(torch.tensor(loss_fct.ignore_index).type_as(labels))

    active_labels = torch.where(
        active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
    )
    print(active_labels)
    # sm_loss = loss_fct(sm_scores.view(-1, self.cls.Phonetic_relationship.pinyin.sm_size), active_labels)
    # print(sm_loss)


def fun():
    a = torch.ones(2, 4)
    print(a.int())


if __name__ == '__main__':
    # test_cude()
    # make_data()
    # align()
    fun()
    # dele()
