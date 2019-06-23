import numpy as np
import os
from tqdm import tqdm


"""
base_dir___1001___"train_0.jpg"
        |      |__"train_1.jpg"
        |__1002___"train_0.jpg"
        ただしラベルが数字のとき
"""

#ベースディレクトリ
base_dir = "mnist"
#作るテキストファイルの名前
path_text = "image_path.txt"
triplet_index_text = "triplet_index_text.txt"
n_plus_1_index_text = "n_plus_1_index_text.txt"
n_pair_index_text = "n_pair_index.txt"


#クラスのディレクトリにそれぞれの画像が入っているとき、すべてのパスが書いてあるtxtとtriplet samplingのインデックスが連なるtxtを作成
def triplet_sampling(base_dir, path_text, triplet_index_text):
    labels = os.listdir(base_dir)
    label_names = []
    for label in tqdm(labels):
        images = os.path.join(base_dir, label)
        for im_name in os.listdir(images):
            path = os.path.join(label, im_name)
            with open(path_text, mode='a') as f:
                f.write("{}\n".format(path))
            label_names.append(int(label))

    label_names = np.array(label_names)
    for _ in tqdm(range(10000)):
        anchor_index = np.random.choice(len(label_names)-1)
        same_lavel_indexs = np.where(label_names==label_names[anchor_index])[0]
        p_index = np.random.choice(same_lavel_indexs)
        n_indexes = np.concatenate([np.arange(same_lavel_indexs[0]),np.arange(same_lavel_indexs[-1]+1, len(label_names))])
        n_index = np.random.choice(n_indexes)
        with open(triplet_index_text, mode='a') as f:
            f.write("{} {} {}\n".format(anchor_index, p_index, n_index))



# (n+1)tuplet
def n_plus_1_sampling(base_dir, path_text, triplet_index_text, epoch_number, N):
    labels = os.listdir(base_dir)
    label_names = []
    for label in tqdm(labels):
        images = os.path.join(base_dir, label)

        for im_name in os.listdir(images):
            path = os.path.join(label, im_name)
            label_names.append(int(label))
            if not os.path.exists(path_text):
                with open(path_text, mode='a') as f:
                    f.write("{}\n".format(path))

    label_names = np.array(label_names)
    for _ in tqdm(range(epoch_number)):
        a_index = np.random.choice(len(label_names)-1)
        same_lavel_indexs = np.where(label_names==label_names[a_index])[0]
        p_index = np.random.choice(same_lavel_indexs)
        n_indexes = np.concatenate([np.arange(same_lavel_indexs[0]),np.arange(same_lavel_indexs[-1]+1, len(label_names))])
        n_indexes = np.random.choice(n_indexes, N, replace=False)
        with open(triplet_index_text, mode='a') as f:
            f.write("{} {} ".format(a_index, p_index))
            for n_index in n_indexes:
                f.write("{} ".format(n_index))
            f.write("\n")


#Nはミニバッチに使うクラス数
#よってNpairなので1バッチに付き使う画像は2N個
#だが結果的にそれを組み合わせて使うデータ数はN個
#N-pair Lossの論文に全部書いてある
def n_pair_sampling(base_dir, path_text, n_pair_index_text, epoch_number, N):
    labels = os.listdir(base_dir)
    label_names = []
    for label in tqdm(labels):
        images = os.path.join(base_dir, label)
        for im_name in os.listdir(images):
            path = os.path.join(label, im_name)
            label_names.append(int(label))
            if not os.path.exists(path_text):
                with open(path_text, mode='a') as f:
                    f.write("{}\n".format(path))
    print(len(label_names))
    label_names = np.array(label_names)
    for _ in tqdm(range(epoch_number)):
        pair_samples = []
        categories = [int(i) for i in os.listdir(base_dir)]
        select_classes = np.random.choice(categories, N, replace=False)
        for select_class in select_classes:
            pair_sample = np.random.choice(np.where(label_names==select_class)[0], 2, replace=False)
            #[x1, x2]
            pair_samples.append(pair_sample)
        pair_samples = np.array(pair_samples)
        # print("pair", pair_samples)
        anchors = pair_samples[:,0]
        positives = pair_samples[:,1]
        # print("anchors", anchors,"positives" , positives)
        with open(n_pair_index_text, mode='a') as f:
            for anchor_index in anchors:
                f.write("{} ".format(anchor_index))
            f.write(",")
            for postive_index in positives:
                f.write("{} ".format(postive_index))
            f.write("\n")
        #1970 16461 19268 2167 17340 802 1674 17662 11609 17526 6811 18843 13677 18314 7386 141 20243 11134 7919 4449 ,1940 16456 19257 2198 17318 743 1660 17595 11607 17529 6706 18824 13667 18350 7377 158 20238 11160 7931 4439




#クラスの個数とラベルの個数を数える
def label_count(base_dir):
    labels = os.listdir(base_dir)
    label_names = []
    label_count = {}
    for label in tqdm(labels):
        images = os.path.join(base_dir, label)
        for im_name in os.listdir(images):
            path = os.path.join(label, im_name)
            with open(path_text, mode='a') as f:
                f.write("{}\n".format(path))
            label_names.append(int(label))

            if label not in label_count.keys():
                label_count[label] = 0
            label_count[label] += 1
    print("all_class:{}個".format(len(labels)))
    print("all_img{}枚".format(sum(label_count.values())))

    # print(len(label_count.values()))
    print("min_number:", min(label_count.values()))
    print("max_number:", max(label_count.values()))

if __name__ == "__main__":
    n_pair_sampling(base_dir, path_text, n_pair_index_text, epoch_number=2000 , N=5)
    # n_plus_1_sampling(base_dir, path_text, n_plus_1_index_text, epoch_number=10000 , N=5)
    # label_count(base_dir)
