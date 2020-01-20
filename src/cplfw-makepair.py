"""
CALFWのペアファイル作成
・1つのID-2画像ずつのリストを作成（一致/不一致同じ画像使用）
・10セット交差検証用にペアファイルを作成する
"""
import argparse
import sys
import glob
import os

def main(args):
    """
    1つのID-2画像ずつのリストを作成
    :param args: 入力値（KCCSアライン画像）
    :return: 1つのID-2画像ずつのリスト。
    """
    # ファイル名読み込んでリスト化(例:[[Aaron_Eckyhart 0001][Abdullah 0001],,,])
    image_list = []
    all_image_list = []
    for filename in glob.glob(args.align_images + "/*"):
        image = (os.path.splitext(os.path.basename(filename))[0])
        image = image.split("_")
        image_name = '_'.join([str(i) for i in image[0:-1]])
        image_num = image[-1]
        image_list.append(image_name)
        image_list.append(image_num)
        all_image_list.append(image_list)
        image_list = []
    print("len(all_image_list)")
    print(len(all_image_list))

    # ファイル名リストをIDごとに多次元化
    id_list = []
    all_id_list = []
    id = all_image_list[0][0]
    for line in all_image_list:
        if line[0] == id:
            id_list.append(line)
        else:
            id = line[0]
            all_id_list.append(id_list)
            id_list = []
            id_list.append(line)
    all_id_list.append(id_list)
    print("len(all_id_list)")
    print(len(all_id_list))

    # all_id_listから2枚/1IDに絞る
    count = 0
    id_list = []
    pair_id_list = []
    for id in all_id_list:
        if len(id) > 1:
            for image in id:
                count = count + 1
                if count == 1:
                    id_list.append(image)
                elif count == 2:
                    id_list.append(image)
                    pair_id_list.append(id_list)
                    count = 0
                    id_list = []
                    break
        else:
            print("1枚だけのID")
            print(id)
    print("len(pair_id__list)")
    print(len(pair_id_list))

    # ペアリスト作成
    pairs = make_pairfile(pair_id_list,args.align_images)

    #pairファイルとして書き込み
    with open(args.pair_file, 'wt') as f:
        f.write("root1\troot2\timage1\troot1\troot2\timage2\tpositive\n")
        for line in pairs:
            tab_line = '\t'.join([str(i) for i in line])
            f.write(tab_line + "\n")

def make_pairfile(pair_id_list,align_images):
    """
    ペアファイルリスト作成
    :param image2_list: 2枚/IDのリスト
    :param align_images：アライン画像までのパス
    :return: ペアリスト
    """
    set_num = len(pair_id_list) // 10
    itti_list = []
    huitti_list = []
    pair = []
    pairs = []
    # root1:calfw_mtcnnpyp_160_32
    # root2:images
    head,tail = os.path.split(align_images)
    _, root1 = os.path.split(head)
    root2 = tail
    for i in range(10):
        set_images = pair_id_list[i*set_num:(i*set_num)+set_num]
        #一致ペア作成
        for id in range(set_num):
            pair.extend([root1,root2])
            pair.append('_'.join([str(i) for i in set_images[id][0][0:2]]))
            pair.extend([root1, root2])
            pair.append('_'.join([str(i) for i in set_images[id][1][0:2]]))
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
                pair.append('_'.join([str(i) for i in set_images[id][0][0:2]]))
                pair.extend([root1, root2])
                pair.append('_'.join([str(i) for i in set_images[0][1][0:2]]))
                pair.append(0)
                huitti_list.append(pair)
                pair = []
            else:
                pair.extend([root1,root2])
                pair.append('_'.join([str(i) for i in set_images[id][0][0:2]]))
                pair.extend([root1, root2])
                pair.append('_'.join([str(i) for i in set_images[id+1][1][0:2]]))
                pair.append(0)
                huitti_list.append(pair)
                pair = []
        print(str(len(huitti_list)) + ":" + str(i))
        pairs.extend(huitti_list)
        huitti_list = []
    print("len(pairs)")
    print(len(pairs))
    return pairs


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('align_images', type=str,
                        help='Aligned images.(D:\l4f\dataset\public\calfw\calfw_mtcnnpyp_160_32\images)')
    parser.add_argument('pair_file', type=str,
                        help='例：D:\\共有フォルダ\\【一時使用】山田\\L4F\\dataset\\calfw\\pair\\pair_calfw.txt)')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))