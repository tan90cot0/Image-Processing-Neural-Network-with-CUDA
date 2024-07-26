import cv2
import numpy as np
import os

np.set_printoptions(linewidth=np.inf,formatter={'float': '{: 0.6f}'.format})

def img_to_array (file_path):
    img = cv2.imread("img/" + file_path, 0)
    if img.shape != [28, 28]:
        img2 = cv2.resize(img,(28, 28))
        
    img = img2.reshape(28, 28, -1);

    #revert the image,and normalize it to 0-1 range
    #img = 1.0 - img/255.0
    img = img/255.0
    return np.ravel(np.array(img, 'float32'))

def main ():
    directory = os.fsencode("img/")
    with open("pre-proc-img/input.dat", "wb") as f:
        inp = np.array([], 'float32')
        for file in os.listdir(directory):
            filename = os.fsdecode(file)
            arr = img_to_array(filename)
            #for x in arr:
            #    f.write(str(x) + ' ')
            #f.write('\n')
            inp = np.concatenate([inp, arr])
            with open("output/gold_label.txt", "a") as g:
                g.write(filename[10] + '\n')
            with open("output/gold_file.txt", "a") as g:
                g.write(filename + '\n')
        inp.tofile(f)

    with open("weights/conv1.txt", "r") as f:
        arr = []
        for line in f.readlines():
            arr.append(float(line))
        output_file = open("weights/conv1.dat", 'wb')
        arr = np.array(arr, 'float32')
        arr.tofile(output_file)
        output_file.close()

    with open("weights/conv2.txt", "r") as f:
        arr = []
        for line in f.readlines():
            arr.append(float(line))
        output_file = open("weights/conv2.dat", 'wb')
        arr = np.array(arr, 'float32')
        arr.tofile(output_file)
        output_file.close()

    with open("weights/fc1.txt", "r") as f:
        arr = []
        for line in f.readlines():
            arr.append(float(line))
        output_file = open("weights/fc1.dat", 'wb')
        arr = np.array(arr, 'float32')
        arr.tofile(output_file)
        output_file.close()

    with open("weights/fc2.txt", "r") as f:
        arr = []
        for line in f.readlines():
            arr.append(float(line))
        output_file = open("weights/fc2.dat", 'wb')
        arr = np.array(arr, 'float32')
        arr.tofile(output_file)
        output_file.close()
        

if __name__ == "__main__":
    main()
