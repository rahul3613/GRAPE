import argparse

def add_uji_subparser(subparsers):
    subparser = subparsers.add_parser('uji')
    # mc settings
    subparser.add_argument('--domain', type=str, default='uji')
    subparser.add_argument('--data', type=str, default='UJIndoorLoc')
    subparser.add_argument('--train_edge', type=float, default=0.8)
    subparser.add_argument('--split_sample', type=float, default=0.)
    subparser.add_argument('--split_by', type=str, default='y') # 'y', 'random'
    subparser.add_argument('--split_train', action='store_true', default=False)
    subparser.add_argument('--split_test', action='store_true', default=False)
    subparser.add_argument('--train_y', type=float, default=0.8)
    subparser.add_argument('--node_mode', type=int, default=0)  # 0: feature onehot, sample all 1; 1: all onehot