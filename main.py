import argparse
import yaml
from train import Trainer
from test import Tester

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--env', type=str, default='BoxingNoFrameskip-v4',
                   help='BoxingNoFrameskip-v4 | BreakoutNoFrameskip-v4')
    p.add_argument('--mode', type=str, default='train', help='train | test')
    p.add_argument('--config', type=str, default='config/dqn.yaml')
    args = p.parse_args()

    # get hyper-parameters file
    config_dir = 'config/dqn.yaml'
    with open(config_dir, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # set that env
    config['env'] = args.env

    if args.mode == 'train':
        trainer = Trainer(config)
        trainer.train()
    elif args.mode == 'test':
        tester = Tester(config)
        tester.test()
    else:
        raise NotImplementedError
