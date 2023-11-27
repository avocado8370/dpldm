import argparse
import os
import shutil
import time

from omegaconf import OmegaConf

from train.encode import encode
from train.sample import sample
from train.train_ae import train_ae
from train.train_ddpm import train_ddpm
from train.train_ncsnpp import train_ncsnpp

if __name__ == '__main__':
    parser = argparse.ArgumentParser("DPLDM")
    # 添加子命令
    subparsers = parser.add_subparsers(dest='subcommand', help='Available subcommands')

    # 子命令 train1
    train1_parser = subparsers.add_parser('train1', help='train first stage')
    train1_parser.add_argument('--config', type=str, help='location of config file')

    # 子命令 encode
    encode_parser = subparsers.add_parser('encode', help='generate encoded dataset')
    encode_parser.add_argument('--config', type=str, help='location of the config file used to train first stage model')
    encode_parser.add_argument('--path', type=str, help='state dict files of first stage model')
    encode_parser.add_argument('--target-path', type=str, help='target path of encoding dataset')
    encode_parser.add_argument('--labeled', type=str, help='if labeled')
    # 子命令 train2
    train2_parser = subparsers.add_parser('train2', help='Description for train2 subcommand')
    train2_parser.add_argument('--path', type=str, help='state dict files of first stage model')
    train2_parser.add_argument('--config-stage1', type=str, help='location of the config file used to train first stage'
                                                                 'model')
    train2_parser.add_argument('--config', type=str, help='location of the config file used to train diffusion model')

    # 子命令 sample
    sample_parser = subparsers.add_parser('sample', help='Description for sample subcommand')
    sample_parser.add_argument('--path', type=str, help='state dict files of first stage model')
    sample_parser.add_argument('--config-stage1', type=str, help='location of the config file used to train first stage'
                                                                 'model')
    sample_parser.add_argument('--denoisez-path', type=str, help='denoisez pkl file path')
    sample_parser.add_argument('--output', type=str, help='sample results path')

    parser.add_argument('--config', help='配置文件名称')
    args = parser.parse_args()

    if args.subcommand == 'train1':
        print('Running train1 with argument:', args.config)
        conf = OmegaConf.load(f'{args.config}')
        config_file = f'{args.config}'
        destination_folder = f'{conf.output_dir}'
        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        ts = time.time()
        timeStr = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
        output_dir = f'{destination_folder}/{timeStr}'
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        shutil.copy(config_file, output_dir)
        train_ae(conf, output_dir)

    elif args.subcommand == 'encode':
        print('Running encode with argument:', args.config)
        conf = OmegaConf.load(f'{args.config}')
        path = args.path
        target_path = args.target_path
        labeled = args.labeled
        encode(conf, path, target_path, labeled)
    elif args.subcommand == 'train2':
        print('Running train2 with argument:', args.config)
        conf_stage1 = OmegaConf.load(f'{args.config_stage1}')
        conf = OmegaConf.load(f'{args.config}')
        path = args.path
        train_ncsnpp(conf_stage1, conf, path)
    elif args.subcommand == 'sample':
        print('Running sample with argument:', args.config)
        conf_stage1 = OmegaConf.load(f'{args.config_stage1}')
        path = args.path
        output = args.output
        denoisez_path = args.denoisez_path
        sample(denoisez_path, path, conf_stage1, output)


    else:
        print('Invalid subcommand')
