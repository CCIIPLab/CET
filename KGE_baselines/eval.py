import argparse
from tools import eval


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='../data/FB15kET/merge/')
    parser.add_argument('--model_path', type=str, default='./ckpts/TransE_l1_FB15kET_0')
    parser.add_argument('--dataset', type=str, default='FB15kET')
    parser.add_argument('--model', type=str, default='TransE_l1')
    parser.add_argument('--gamma', type=float, default=0.0)
    args, _ = parser.parse_known_args()
    return args


if __name__ == '__main__':
    args = get_params()
    prefix = '%s_%s' % (args.dataset, args.model)
    res = eval(args.data_path, args.model_path, prefix, args.model, final_eval=True, gamma=args.gamma)
    for metric in res:
        print('%s: %f' % (metric, res[metric]))
