# simple-cinic-10-dataset-loader
Like the title...

You can get the cinic-10 dataset follow this link<br>
```
https://datashare.is.ed.ac.uk/bitstream/handle/10283/3192/CINIC-10.tar.gz
```

This loader does not include the download function, so you need to use other tools to download the original dataset and extract it..<br>
Find the folder containing train, valid, test folder. (dataset root folder)<br>

You can load dataset from dataset folder follow<br>
```
ds = cinic10_dataset()
ds.load('datasets/cinic10')
```

then you can save the dataset in npz file to improve next loading speed.<br>
```
ds.save_npz('datasets/cinic10.npz')
```

load from npz file is fast than load from dataset folder.<br>
```
ds.load_from_npz('datasets/cinic10.npz')
```

when you load dataset complete, you can use data() get all image and label.<br>
```
train_x, train_y, valid_x, valid_y, test_x, test_y = ds.data()
```

or you can use some simple api get a data batch.<br>
```
print(ds.get_train_batch_count(100))
print(ds.get_valid_batch_count(100))
print(ds.get_test_batch_count(100))

for b in range(ds.get_train_batch_count(100)):
    ds.get_train_batch(b, 100)

for b in range(ds.get_valid_batch_count(100)):
    ds.get_valid_batch(b, 100)

for b in range(ds.get_test_batch_count(100)):
    ds.get_test_batch(b, 100)
```
