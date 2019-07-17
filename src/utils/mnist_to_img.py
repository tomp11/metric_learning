import os
from PIL import Image
import chainer

def save(data, index, num, mnist_path):
    img = Image.new("L", (28, 28))
    pix = img.load()
    for i in range(28):
        for j in range(28):
            pix[i, j] = int(data[i+j*28]*256)
    filename = os.path.join(mnist_path, str(num), "train{0:04d}.png".format(index))
    img.save(filename)
    print(filename)

def main():
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # "metric_test"
    dataset_path = os.path.join(base, "datasets")
    mnist_path = os.path.join(dataset_path, "mnist")
    print(base)
    print(dataset_path)
    print(mnist_path)

    train, _ = chainer.datasets.get_mnist()

    if not os.path.isdir(dataset_path):
        os.mkdir(dataset_path)
    if not os.path.isdir(mnist_path):
        os.mkdir(mnist_path)
    for i in range(10):
        dirname = str(i)
        if not os.path.isdir(os.path.join(mnist_path, dirname)):
            os.mkdir(os.path.join(mnist_path, dirname))
    for i in range(len(train)*9//10):
        save(train[i][0], i, train[i][1], mnist_path)

if __name__ == '__main__':
    main()
