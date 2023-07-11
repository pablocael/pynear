import os
import wget
from tempfile import TemporaryDirectory

temp = TemporaryDirectory(suffix="datasets")

output = os.path.join(temp.name, "coco_train2017.zip")
wget.download("http://images.cocodataset.org/zips/val2017.zip", out=output)
