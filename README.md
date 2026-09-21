# Custom DataLoader

> Technical-test solution: a custom PyTorch `Dataset`/`DataLoader` for an image classification set labelled from a spreadsheet, plus label inspection and data augmentation scripts.

The dataset is an *éco-compteur* collection — street images of bikes,
pedestrians, and scooters — where labels live in an Excel file rather than in
the directory structure. That rules out `ImageFolder` and makes a custom
`Dataset` the right answer.

## The scripts

| Script | Purpose |
|---|---|
| [`PropreDataloder.py`](PropreDataloder.py) | The `PropreDataset` class — reads the annotation spreadsheet, loads images, applies random padding, returns `(image, label)` tensors ready for a `DataLoader` |
| [`Display_Img_Lab.py`](Display_Img_Lab.py) | Sanity check — picks a random row, displays the image with its decoded class name |
| [`Techniques_data_augmentation.py`](Techniques_data_augmentation.py) | Side-by-side visualisation of augmentation transforms: manual padding, random rotation, flips |

`Dataset_housing_price.zip` is an unrelated dataset kept here for reuse — it's
the download source for the ingestion stage of
[Project-Mlops](https://github.com/Fezzaioussama/Project-Mlops).

## Classes

Labels are seven columns in the spreadsheet, decoded via `argmax`:

`m-loc` · `e-loc` · `meca` · `elec` · `nn_id` · `trot-loc` · `trot`

(rental and private mechanical/electric bikes, unidentified, rental and private
scooters).

## `PropreDataset`

The class implements the standard three methods:

- `__len__` — row count of the dataframe.
- `__getitem__(idx)` — resolves the image filename from column 1, loads it from
  `root_dir`, applies random padding then any configured transform, and returns
  the image with its label tensor.
- `random_padding(image, max_padding)` — pads by a random amount in
  `[0, max_padding]` with black borders, so the subject sits at a varying offset
  each epoch.

Random padding is the interesting choice here: it's augmentation that shifts the
subject's position within the frame without cropping anything out, which suits a
counter dataset where objects appear at varying distances from the camera.

## Running

```bash
pip install torch torchvision pandas numpy matplotlib pillow openpyxl
python Display_Img_Lab.py
```

> **Paths are hardcoded** to a Windows drive
> (`D:\Python_code\datasetecocompteur`), with one Colab Drive path left in
> `Techniques_data_augmentation.py`. Point `file_path` and `root_dir` at your
> own copy before running anything. The dataset itself is not in this repo.

## Usage sketch

```python
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from PropreDataloder import PropreDataset

df = pd.read_excel("annotations.xlsx")
dataset = PropreDataset(
    dataframe=df,
    root_dir="path/to/images",
    ttransform=transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]),
    max_padding=20,
)
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

Note the constructor argument is `ttransform`, not `transform`.
