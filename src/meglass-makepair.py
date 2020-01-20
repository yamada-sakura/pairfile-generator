"""
ペアファイル作成
・1つのID-4画像ずつのリストを作成（一致/不一致同じ画像使用）
・10セット交差検証用にペアファイルを作成する
"""
import argparse
import sys
import glob
import os

def main(args):
    """
    1つのID-4画像ずつのリストを作成
    :param args: 入力値（公式アライン画像、ラベル付き画像リスト(meta.txt)）
    :return: 1つのID-4画像ずつのリスト。めがね[なし、なし、あり、あり]
    """
    # metaファイル読み込んでリスト化(例:[10032527 N08_identity_4 2582182573_0.png 0])
    meta_list = []
    with open(args.meta_file, 'r') as f:
        for line in f.readlines()[:]:
            line_2_split = line.split()
            name, _ = os.path.splitext(line_2_split[0])
            line_4_split = name.split('@')
            line_4_split.append(line_2_split[1])
            meta_list.append(line_4_split)

    # アラインできたものだけに絞る
    align_list = []
    for image in meta_list:
        path = args.align_images + "\\" + ('@'.join([str(i) for i in image[0:3]])) + '.png'
        print(path)
        if os.path.isfile(path):
            align_list.append(image)

    # metaリストをIDごとに多次元化
    id_list = []
    id_meta_list = []
    id_1 = align_list[0][0]
    id_2 = align_list[0][1]
    for line in align_list:
        if line[0] == id_1 and line[1] == id_2:
            id_list.append(line)
        else:
            id_1 = line[0]
            id_2 = line[1]
            id_meta_list.append(id_list)
            id_list = []
            id_list.append(line)
    id_meta_list.append(id_list)

    # id_meta_listから4枚/1IDに絞る
    zero_count = 0
    one_count = 0
    id_list = []
    image_list = []
    for id in id_meta_list:
        for image in id:
            if image[3] == "0" and zero_count != 2:
                id_list.insert(0,image)
                zero_count = zero_count + 1
            elif image[3] == "1" and one_count != 2:
                id_list.append(image)
                one_count = one_count + 1
            if zero_count == 2 and one_count == 2:
                image_list.append(id_list)
                zero_count = 0
                one_count = 0
                id_list = []
                break
        # アラインできず4枚以下だったID
        if zero_count < 2 or one_count < 2:
            zero_count = 0
            one_count = 0
            id_list = []

    # ペアリスト作成
    pairs = make_pairfile(image_list,args.align_images)

    #pairファイルとして書き込み
    with open(args.pair, 'wt') as f:
        f.write("root1\troot2\timage1\troot1\troot2\timage2\tpositive\n")
        for line in pairs:
            tab_line = '\t'.join([str(i) for i in line])
            f.write(tab_line + "\n")

def make_pairfile(image_list,align_images):
    """
    ペアリスト作成
    :param image_list: 4枚/IDのリスト
    :param align_images：アライン画像までのパス
    :return: ペアリスト
    """
    set_num = len(image_list) // 10
    print("set_num")
    print(set_num)
    itti_list = []
    huitti_list = []
    pair = []
    pairs = []
    head,tail = os.path.split(align_images)
    _, root1 = os.path.split(head)
    root2 = tail
    for i in range(10):
        set_images = image_list[i*set_num:(i*set_num)+set_num]
        #一致ペア作成
        for id in range(set_num):
            # 1ペア目
            pair.extend([root1,root2])
            pair.append('@'.join([str(i) for i in set_images[id][2][0:3]]))
            pair.extend([root1, root2])
            pair.append('@'.join([str(i) for i in set_images[id][0][0:3]]))
            pair.append(1)
            itti_list.append(pair)
            pair = []
            # 2ペア目
            pair.extend([root1, root2])
            pair.append('@'.join([str(i) for i in set_images[id][3][0:3]]))
            pair.extend([root1, root2])
            pair.append('@'.join([str(i) for i in set_images[id][1][0:3]]))
            pair.append(1)
            itti_list.append(pair)
            pair = []
        print(str(len(itti_list)) + ":" + str(i))
        pairs.extend(itti_list)
        itti_list = []
        #不一致ペア作成
        for id in range(set_num):
            if id == (set_num-1):
                pair.extend([root1,root2])
                pair.append('@'.join([str(i) for i in set_images[id][2][0:3]]))
                pair.extend([root1, root2])
                pair.append('@'.join([str(i) for i in set_images[0][0][0:3]]))
                pair.append(0)
                huitti_list.append(pair)
                pair = []
                pair.extend([root1, root2])
                pair.append('@'.join([str(i) for i in set_images[id][3][0:3]]))
                pair.extend([root1, root2])
                pair.append('@'.join([str(i) for i in set_images[0][1][0:3]]))
                pair.append(0)
                huitti_list.append(pair)
                pair = []
            else:
                pair.extend([root1,root2])
                pair.append('@'.join([str(i) for i in set_images[id][2][0:3]]))
                pair.extend([root1, root2])
                pair.append('@'.join([str(i) for i in set_images[id+1][0][0:3]]))
                pair.append(0)
                huitti_list.append(pair)
                pair = []
                pair.extend([root1, root2])
                pair.append('@'.join([str(i) for i in set_images[id][3][0:3]]))
                pair.extend([root1, root2])
                pair.append('@'.join([str(i) for i in set_images[id + 1][1][0:3]]))
                pair.append(0)
                huitti_list.append(pair)
                pair = []
        print(str(len(huitti_list)) + ":" + str(i))
        pairs.extend(huitti_list)
        huitti_list = []
    return pairs


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('meta_file', type=str,
                        help='meta_file of all images.(D:\l4f\dataset\public\meglass\meta.txt)')
    parser.add_argument('align_images', type=str,
                        help='Aligned images.(D:\l4f\dataset\public\meglass\MeGlass_mtcnnpyp_160_32\meglass_120)')
    parser.add_argument('pair', type=str,
                        help='書き込むペアファイル')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))