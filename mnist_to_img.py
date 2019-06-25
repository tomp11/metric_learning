import os
from PIL import Image
import chainer

def save(data, index, num):
    img = Image.new("L", (28, 28))
    pix = img.load()
    for i in range(28):
        for j in range(28):
            pix[i, j] = int(data[i+j*28]*256)
    filename ="mnist_metiric/" + str(num) + "/test" + "{0:04d}".format(index) + ".png"
    img.save(filename)
    print(filename)

def main():
    _, test = chainer.datasets.get_mnist()
    for i in range(10):
        dirname = str(i)
        if not os.path.isdir("mnist"):
            os.mkdir("mnist")
        if not os.path.isdir(os.path.join("mnist", dirname)):
            os.mkdir(os.path.join("mnist", dirname))
    for i in range(len(test)):
        save(test[i][0], i, test[i][1])

if __name__ == '__main__':
    main()
