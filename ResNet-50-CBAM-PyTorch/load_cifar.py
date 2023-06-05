import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

data: dict = unpickle("/nfs/stak/users/morgamat/hpc-share/CS_499/CS_499_Term_Project/ResNet-50-CBAM-PyTorch/cifar-10-batches-py/data_batch_1")
# Keys: b'batch_label', b'labels', b'data', b'filenames']
print(data[b'batch_label'])
print(len(data[b'labels']), 'labels')
print(len(data[b'data']), 'data')
print(len(data[b'filenames']), 'filenames')

i = 0
for img in data[b'data']:
    i += 1
    single_img_reshaped = np.reshape(img, (32, 32, 3))
    print(img[:10])
    print(single_img_reshaped[:1])
    file_path = 'line_image.png'  # Specify the file path and extension
    plt.imshow(single_img_reshaped)

    if i > 7:
        break

    break







# labels = unpickle("/nfs/stak/users/morgamat/hpc-share/CS_499/CS_499_Term_Project/ResNet-50-CBAM-PyTorch/cifar-10-batches-py/batches.meta")
# print(labels)






class_names = ['airplane',
'automobile',
'bird',
'cat',
'deer',
'dog',
'frog',
'horse',
'ship',
'truck']
nb_classes = len(class_names)
