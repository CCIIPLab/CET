import torch
import logging
import os
import dgl

def set_logger(args):
    if not os.path.exists(os.path.join(args['save_dir'], args['dataset'])):
        os.makedirs(os.path.join(os.getcwd(), args['save_dir'], args['dataset']))

    log_file = os.path.join(args['save_dir'], args['dataset'], 'log.txt')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.DEBUG,
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )

    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def read_id(path):
    tmp = dict()
    with open(path, encoding='utf-8') as r:
        for line in r:
            e, t = line.strip().split('\t')
            tmp[e] = int(t)
    return tmp


def load_triple(path, e2id, r2id):
    head = []
    e_type = []
    tail = []
    with open(path, encoding='utf-8') as r:
        for line in r:
            h, r, t = line.strip().split('\t')
            h, r, t = e2id[h], r2id[r], e2id[t]
            head.append(h)
            e_type.append(r)
            tail.append(t)
    return head, e_type, tail


def load_ET(path, e2id, t2id, r2id):
    head = []
    e_type = []
    tail = []
    with open(path, encoding='utf-8') as r:
        for line in r:
            e, t = line.strip().split('\t')
            e, t = e2id[e], t2id[t] + len(e2id)
            head.append(e)
            tail.append(t)
            e_type.append(r2id['type'])
    return head, e_type, tail


def load_labels(paths, e2id, t2id):
    labels = torch.zeros(len(e2id), len(t2id))
    for path in paths:
        with open(path, encoding='utf-8') as r:
            for line in r:
                e, t = line.strip().split('\t')
                e_id, t_id = e2id[e], t2id[t]
                labels[e_id, t_id] = 1
    return labels


def load_id(path, e2id):
    ret = set()
    with open(path, encoding='utf-8') as r:
        for line in r:
            e, t = line.strip().split('\t')
            ret.add(e2id[e])
    return list(ret)


def load_graph(data_dir, e2id, r2id, t2id, loadET=True, loadKG=True):
    # load graph with input features, labels and edge type
    train_label = load_labels([os.path.join(data_dir, 'ET_train.txt')], e2id, t2id)
    train_id = train_label.sum(1).nonzero().squeeze()
    valid_id = load_id(os.path.join(data_dir, 'ET_valid.txt'), e2id)
    test_id = load_id(os.path.join(data_dir, 'ET_test.txt'), e2id)
    all_true = load_labels([
        os.path.join(data_dir, 'ET_train.txt'),
        os.path.join(data_dir, 'ET_valid.txt'),
        os.path.join(data_dir, 'ET_test.txt'),
    ], e2id, t2id).half()
    if loadKG:
        head1, e_type1, tail1 = load_triple(os.path.join(data_dir, 'train.txt'), e2id, r2id)
    else:
        head1, e_type1, tail1 = [], [], []
    if loadET:
        head2, e_type2, tail2 = load_ET(os.path.join(data_dir, 'ET_train.txt'), e2id, t2id, r2id)
    else:
        head2, e_type2, tail2 = [], [], []

    head = torch.LongTensor(head1 + head2 + tail1 + tail2)
    tail = torch.LongTensor(tail1 + tail2 + head1 + head2)
    g = dgl.graph((head, tail))

    e_type1 = torch.LongTensor(e_type1)
    e_type2 = torch.LongTensor(e_type2)
    e_type = torch.cat([e_type1, e_type2, e_type1 + len(r2id), e_type2 + len(r2id)], dim=0)
    g.edata['etype'] = e_type
    if loadET:
        g.ndata['id'] = torch.arange(len(e2id) + len(t2id))
    else:
        g.ndata['id'] = torch.arange(len(e2id))

    return g, train_label, all_true, train_id, valid_id, test_id


def evaluate(path, predict, all_true, e2id, t2id):
    logs = []
    f = open('./rank.txt', 'w', encoding='utf-8')
    with open(path, 'r', encoding='utf-8') as r:
        for line in r:
            e, t = line.strip().split('\t')
            e, t = e2id[e], t2id[t]
            tmp = predict[e] - all_true[e]
            tmp[t] = predict[e, t]
            argsort = torch.argsort(tmp, descending=True)
            ranking = (argsort == t).nonzero()
            assert ranking.size(0) == 1
            ranking = ranking.item() + 1
            print(line.strip(), ranking, file=f)
            logs.append({
                'MRR': 1.0 / ranking,
                'MR': float(ranking),
                'HIT@1': 1.0 if ranking <= 1 else 0.0,
                'HIT@3': 1.0 if ranking <= 3 else 0.0,
                'HIT@10': 1.0 if ranking <= 10 else 0.0
            })
    MRR = 0
    for metric in logs[0]:
        tmp = sum([_[metric] for _ in logs]) / len(logs)
        if metric == 'MRR':
            MRR = tmp
        logging.debug('%s: %f' % (metric, tmp))
    return MRR


def cal_loss(predict, label, beta):
    loss = torch.nn.BCELoss(reduction='none')
    output = loss(predict, label)
    positive_loss = output * label
    negative_weight = predict.detach()
    negative_weight = beta * (negative_weight - negative_weight.pow(2)) * (1 - label)
    negative_loss = negative_weight * output
    return positive_loss.mean(), negative_loss.mean()
