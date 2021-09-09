import numpy as np
import torch
import os
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


def read_triple(path, e2id, r2id):
    triples = []
    with open(path) as r:
        for l in r:
            h, r, t = l.strip().split('\t')
            triples.append((e2id[h], r2id[r], e2id[t]))
    return triples


# test data loader
class TestDataset(Dataset):
    def __init__(self, triples, all_true_triples, types):
        self.len = len(triples)
        self.triple_set = set(all_true_triples)
        self.triples = triples
        self.types = types

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        head, relation, tail = self.triples[idx]

        bias = []
        negative_sample = []
        cnt = 0
        target = 0
        for rand_type in self.types:
            if rand_type == tail:
                bias.append(0)
                negative_sample.append(tail)
                target = cnt
            elif (head, relation, rand_type) in self.triple_set:
                bias.append(-10)
                negative_sample.append(tail)
            else:
                bias.append(0)
                negative_sample.append(rand_type)
            cnt += 1

        bias = torch.LongTensor(bias)
        negative_sample = torch.LongTensor(negative_sample)
        positive_sample = torch.LongTensor((head, relation, tail))

        return positive_sample, negative_sample, bias, target

    @staticmethod
    def collate_fn(data):
        positive_sample = torch.stack([_[0] for _ in data], dim=0)
        negative_sample = torch.stack([_[1] for _ in data], dim=0)
        bias = torch.stack([_[2] for _ in data], dim=0)
        target = torch.LongTensor([_[3] for _ in data])
        return positive_sample, negative_sample, bias, target


# score function
def ComplEx(h, r, t):
    re_h, im_h = torch.chunk(h, 2, dim=-1)
    re_r, im_r = torch.chunk(r, 2, dim=-1)
    re_t, im_t = torch.chunk(t, 2, dim=-1)
    re_hr = re_h * re_r - im_h * im_r
    im_hr = re_h * im_r + im_h * re_r
    score = re_hr * re_t + im_hr * im_t
    return score.sum(dim=-1)


def TransE(h, r, t):
    score = h + r - t
    score = score.norm(p=1, dim=-1)
    return -score


def RotatE(h, r, t, gamma):
    re_head, im_head = torch.chunk(h, 2, dim=-1)
    re_tail, im_tail = torch.chunk(t, 2, dim=-1)

    phase_rel = r / (((gamma+2)/200) / np.pi)
    re_rel, im_rel = torch.cos(phase_rel), torch.sin(phase_rel)
    re_score = re_head * re_rel - im_head * im_rel
    im_score = re_head * im_rel + im_head * re_rel
    re_score = re_score - re_tail
    im_score = im_score - im_tail
    score = torch.stack([re_score, im_score], dim=0)
    score = score.norm(dim=0)
    return -score.sum(-1)


def eval(data_path, model_path, prefix, model, gamma, final_eval=True):
    entity_path = os.path.join(model_path, prefix + '_entity.npy')
    relation_path = os.path.join(model_path, prefix + '_relation.npy')
    # load data
    # load e2id & r2id
    e2id = dict()
    r2id = dict()
    with open(data_path + 'entities.tsv') as r:
        for line in r:
            idx, e = line.strip().split('\t')
            e2id[e] = int(idx)
    with open(data_path + 'relations.tsv') as r:
        for line in r:
            idx, r = line.strip().split('\t')
            r2id[r] = int(idx)

    # load type
    types = []
    with open(data_path + 'types.txt') as r:
        for line in r:
            types.append(e2id[line.strip()])

    # load train/valid/test set
    train = read_triple(data_path + 'train.txt', e2id, r2id)
    valid = read_triple(data_path + 'valid.txt', e2id, r2id)
    test = read_triple(data_path + 'test.txt', e2id, r2id)
    all_true_triple = train + valid + test

    print('train:', len(train))
    print('valid:', len(valid))
    print('test:', len(test))
    print('total:', len(all_true_triple))

    if final_eval:
        dataset = TestDataset(test, all_true_triple, types)
    else:
        dataset = TestDataset(valid, all_true_triple, types)

    testset = DataLoader(
        dataset=dataset,
        batch_size=32,
        num_workers=8,
        collate_fn=TestDataset.collate_fn
    )

    # load model
    entity = torch.from_numpy(np.load(entity_path)).cuda()
    relation = torch.from_numpy(np.load(relation_path)).cuda()

    # eval model
    logs = []
    for data in testset:
        positive, negative, bias, target = data
        positive = positive.cuda()
        negative = negative.cuda()
        bias = bias.cuda()
        target = target.cuda()

        batch_size, negative_num = negative.size(0), negative.size(1)

        head = torch.index_select(entity, dim=0, index=positive[:, 0]).unsqueeze(1)
        rel = torch.index_select(relation, dim=0, index=positive[:, 1]).unsqueeze(1)
        tail = torch.index_select(entity, dim=0, index=negative.view(-1)).reshape(batch_size, negative_num, -1)

        if model == 'ComplEx':
            score = ComplEx(head, rel, tail)
        elif model == 'TransE_l1':
            score = TransE(head, rel, tail)
        elif model == 'RotatE':
            score = RotatE(head, rel, tail, gamma)
        else:
            print('NO SUCH MODEL')
            assert 0
        score += bias
        argsort = torch.argsort(score, dim=1, descending=True)

        for i in range(batch_size):
            rank = (argsort[i, :] == target[i]).nonzero()
            assert rank.size(0) == 1

            rank = rank.item() + 1
            logs.append({
                'MRR': 1.0 / rank,
                'MR': float(rank),
                'HITS@1': 1.0 if rank <= 1 else 0.0,
                'HITS@3': 1.0 if rank <= 3 else 0.0,
                'HITS@10': 1.0 if rank <= 10 else 0.0,
            })

    res = dict()
    for metric in logs[0].keys():
        res[metric] = sum([log[metric] for log in logs]) / len(logs)
    return res
