"""
アラインをしたAsianCelebのデータセットの中で、
ID内の枚数が0また1枚のIDをテキストに書き出しています。
"""
import argparse
import glob
import sys

def main(args):
    Min = 1000
    one_count = 0
    zero_count = 0
    zero_one_list = []

    id_folder =args.id_folder
    for i in glob.glob(id_folder + "/**/"):
        number = len(glob.glob(i + "/*"))
        if number == 1:
            one_count += 1
            zero_one_list.append(i)
        if number == 0:
            zero_count += 1
            zero_one_list.append(i)
        if Min > number:
            Min = number

    with open(args.remove_id, 'wt') as f:
        for line in zero_one_list:
            f.write(line + "\n")

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('id_folder', type=str,
                        help='IDまでのパス')
    parser.add_argument('remove_id', type=str,
                        help='除去したいIDを書き込んでいくテキストファイル')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
