import argparse
import os


def get_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='FB15kET')
    args, _ = parser.parse_known_args()
    return args


def remove_unobserved_entity_type(entity_set, type_set, src, dst):
    with open(src, encoding='utf-8') as r:
        lines = r.readlines()
    with open(dst, 'w', encoding='utf-8') as w:
        for line in lines:
            e, t = line.strip().split()
            if e not in entity_set:
                continue
            if t not in type_set:
                continue
            w.write(line)


def convert_ET2triples(src, dst):
    with open(src, encoding='utf-8') as f:
        data = f.readlines()
    with open(dst, 'w', encoding='utf-8') as f:
        for line in data:
            e, t = line.strip().split('\t')
            f.write(f'{e}\thastype\t{t}\n')


def save_id(src, dst):
    with open(dst, 'w', encoding='utf-8') as w:
        for idx, item in enumerate(src):
            w.write(f'{item}\t{idx}\n')


def main(args):
    # collect entity set
    entity = set()
    relation = set()
    with open(os.path.join(args.dataset, 'original/train.txt'), encoding='utf-8') as r:
        for line in r:
            h, r, t = line.strip().split('\t')
            entity.add(h)
            entity.add(t)
            relation.add(r)

    # remove unobserved entities and collect types
    lines = []
    types = set()
    with open(os.path.join(args.dataset, 'original/Entity_Type_train.txt'), encoding='utf-8') as r:
        for line in r:
            e, t = line.strip().split('\t')
            if e in entity:
                lines.append(line)
                types.add(t)

    with open(os.path.join(args.dataset, 'clean/ET_train.txt'), 'w', encoding='utf-8') as w:
        w.writelines(lines)

    # create clean entity_type_valid/test set
    remove_unobserved_entity_type(entity, types,
                                  src=os.path.join(args.dataset, 'original/Entity_Type_valid.txt'),
                                  dst=os.path.join(args.dataset, 'clean/ET_valid.txt'))
    remove_unobserved_entity_type(entity, types,
                                  src=os.path.join(args.dataset, 'original/Entity_Type_test.txt'),
                                  dst=os.path.join(args.dataset, 'clean/ET_test.txt'))

    # copy train.txt to dir clean
    os.system(f'cp {args.dataset}/original/train.txt {args.dataset}/clean/train.txt')

    # save entity, relation, type
    save_id(entity, os.path.join(args.dataset, 'clean/entities.tsv'))
    save_id(relation, os.path.join(args.dataset, 'clean/relations.tsv'))
    save_id(types, os.path.join(args.dataset, 'clean/types.tsv'))

    # create data files for KGE methods
    convert_ET2triples(f'{args.dataset}/clean/ET_train.txt', f'{args.dataset}/merge/train.txt')
    with open(f'{args.dataset}/clean/train.txt', encoding='utf-8') as f:
        data = f.readlines()
    with open(f'{args.dataset}/merge/train.txt', 'a', encoding='utf-8') as f:
        f.writelines(data)
    convert_ET2triples(f'{args.dataset}/clean/ET_valid.txt', f'{args.dataset}/merge/valid.txt')
    convert_ET2triples(f'{args.dataset}/clean/ET_test.txt', f'{args.dataset}/merge/test.txt')

    with open(f'{args.dataset}/merge/types.txt', 'w', encoding='utf-8') as f:
        for t in types:
            f.write(t + '\n')


if __name__ == '__main__':
    args = get_params()
    if args.dataset not in ['FB15kET', 'YAGO43kET']:
        raise ValueError(f'Dataset {args.dataset} is not exist')
    main(args)
