# COCO Instance Segmentation

[![Open in Layer](https://app.layer.ai/assets/badge.svg)](https://app.layer.ai/layer/instance-segmentation/) [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/layerai/examples/blob/main/instance-segmentation/notebooks/demo.ipynb) [![Layer Examples Github](https://badgen.net/badge/icon/github?icon=github&label)](https://github.com/layerai/examples/tree/main/instance-segmentation)

![Segments AI](https://segments.ai/blog/assets/images/panoptic-datasets/2d.png)

In this project, we are going to train an **instance segmentation** model with a [COCO Dataset](https://cocodataset.org/#home) from [Segments AI](https://segments.ai/). 

Segments AI is a platform for computer vision experts to manage, label and improve datasets which supports image, point cloud and test labeling in various formats including the COCO Instance we've used in this project. 


## How to make predictions

You can load and use the trained model from this project easily. First make sure you have the required libraries.

```
!pip install layer torch torchvision Pillow -q
```

Then, you can fetch the model from Layer and make predictions from an image

```python
import requests
import torch
from torchvision import transforms as T
from PIL import Image

import layer
my_model = layer.get_model("layer/instance-segmentation/models/object_detector:4.2").get_train()

img_url = "https://segmentsai-prod.s3.eu-west-2.amazonaws.com/assets/admin-tobias/515b671a-4ce3-4199-91e4-ad53f155935e.jpg"
original_image = Image.open(requests.get(img_url, stream=True).raw)
trans = T.ToTensor()
img = trans(original_image)

my_model.eval()
with torch.no_grad():
    prediction = my_model([img])

Image.fromarray(prediction[0]['masks'][0, 0].mul(255).byte().cpu().numpy())
```

Here is a sample prediction logged during the training of this model:

https://app.layer.ai/layer/instance-segmentation/models/object_detector#prediction

## How to train an instance segmentation model

You can start training your labeled image dataset on Segments AI directly on Layer asily from your terminal. You don't need a GPU on your local. Layer will train the model on Layer infra and register it to your Layer account.

First make sure you have the latest Layer SDK:
```shell
pip install layer -U
```

Then clone [Layer Examples](https://github.com/layerai/examples) repo:
```shell
git clone https://github.com/layerai/examples
cd layerai/examples/instance-segmentation
```

And now, in your terminal just run the following command. Don't forget to replace your options with your own
`[YOUR_SEGMENTS_API_KEY]` and `[YOUR_SEGMENTS_DATASET]`
```shell
python main.py --segments_api_key=[YOUR_SEGMENTS_API_KEY] --dataset=[YOUR_SEGMENTS_DATASET]
```

Layer will fetch your labeled dataset and start training. Run the following command for all parameters:
```shell
python main.py --help
```


## COCO Instance Segmentation Dataset

We used the sidewalk imagery dataset from Segments AI for this project. The images are labeled by Segments AI. We have used this dataset
in COCO Instance format to train our Pytorch model. 

The dataset have 1000 images labeled with 34 categories.

You can take a look at the data here:
https://segments.ai/segments/sidewalk-imagery/samples

## Model

For this project we have used a modified version of [Torchvision Object Detection tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html). The model itself is a modified [maskrcnn_resnet50_fpn](https://pytorch.org/vision/stable/generated/torchvision.models.detection.maskrcnn_resnet50_fpn.html).

We have made some changes in the original tutorial:
- It accepts only a single class. We modified it so that it accepts multiple classes which was the requirement for this specific dataset
- It reads the masks from a sequence of PNGs. With our changes, it supports reading the masks from the COCO Instance json.

### Experiments

We started our **machine learning experiments** with multiple parameters. 
Here is a comparison of the parameters in our best performing models:

https://app.layer.ai/layer/instance-segmentation/models/object_detector?v=4.2&w=5.3&w=4.1#parameters

After some iterations, we have identified that model is doing really a good job on segmenting cars and people but not with the other classes
like flat road, sidewalk, construction objects. We experimented with more parameters, especially `epochs` and `learning rate`. See our `mAP` scores getting better and better every experimentation.

https://app.layer.ai/layer/instance-segmentation/models/object_detector?v=2.1&w=3.1&w=4.1&w=4.2#mAP-bbox

We experimented with adding/removing classes, here is the difference between the `loss` curves in our experiments.

https://app.layer.ai/layer/instance-segmentation/models/object_detector?v=4.2&w=4.2&w=3.1&w=2.1&w=1.3&w=1.2&w=5.2#loss


## References
- https://segments.ai/
- https://cocodataset.org/#home
- https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html