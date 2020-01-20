"""
ペアファイル作成
・1つのID-2画像ずつのリストを作成
・10セット交差検証用にペアファイルを作成する
"""
import argparse
import sys
import glob
import os

def main(args):
    """
    1つのID-2画像ずつのリストを作成
    :param args: 入力値（除外IDファイル、アライン画像）
    :return: 1つのID-2画像ずつのリスト
    """
    #除外したい画像のリストを作成する
    with open(args.remove_id, 'r') as f:
        file = f.read()
        exclud_file = file.split("\n")
    print(exclud_file)

    #1つのIDに2枚のリストを作成する
    image_list = []
    all_image_list = []
    align_images = os.path.expanduser(args.align_images)
    print(align_images)
    align_dir = os.path.basename(align_images)
    print(align_dir)
    count = 0
    for id_only in os.listdir(align_images + "/"):
        if os.path.isdir(os.path.join(align_images, id_only)):
            print(id_only)
            if id_only in exclud_file:
                count += 1
                continue
            else:
                for index, image in enumerate(glob.glob(align_images + "/" + id_only + "/*")):
                    name, _ = os.path.splitext(os.path.basename(image))
                    image_list.append(align_dir)
                    image_list.append(id_only)
                    image_list.append(name)
                    if index == 0:
                        continue
                    elif index == 1:
                        break
                all_image_list.append(image_list)
                image_list = []
    print(count)
    print(len(all_image_list))
    pairs = make_pairfile(all_image_list)

    #pairファイルとして書き込み
    with open(args.pair, 'wt') as f:
        f.write("root1\tperson1\timage1\troot2\timage2\tpositive\n")
        for line in pairs:
            tab_line = '\t'.join([str(i) for i in line])
            f.write(tab_line + "\n")

def make_pairfile(all_image_list):
    """
    ペアファイル作成
    :param all_image_list: 2枚/IDのリスト
    :return: ペアファイル保存
    """
    set_num = len(all_image_list) // 10
    print(set_num)
    itti_list = []
    huitti_list = []
    pair = []
    pairs = []
    for i in range(10):
        set_images = all_image_list[i*set_num:(i*set_num)+set_num]
        #一致ペア作成
        for id in range(set_num):
            pair.extend(set_images[id][0:3])
            pair.extend(set_images[id][3:6])
            pair.append(1)
            itti_list.append(pair)
            pair = []
        print(str(len(itti_list)) + ":" + str(i))
        pairs.extend(itti_list)
        itti_list = []
        #不一致ペア作成
        for id in range(set_num):
            if id == (set_num-1):
                pair.extend(set_images[id][0:3])
                pair.extend(set_images[0][3:6])
                pair.append(0)
            else:
                pair.extend(set_images[id][0:3])
                pair.extend(set_images[id+1][3:6])
                pair.append(0)
            huitti_list.append(pair)
            pair = []
        print(str(len(huitti_list)) + ":" + str(i))
        pairs.extend(huitti_list)
        huitti_list = []
    return pairs


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('remove_id', type=str,
                        help='List of files to be excluded.')
    parser.add_argument('align_images', type=str,
                        help='Aligned images.')
    parser.add_argument('pair', type=str,
                        help='書き込むペアファイル')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))