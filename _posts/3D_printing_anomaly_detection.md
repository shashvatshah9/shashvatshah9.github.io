# Early detection of 3D printing issues
This [Kaggle competition](https://www.kaggle.com/competitions/early-detection-of-3d-printing-issues) required us to focus on a particular kind of anomaly, which was to detect under extrusion. Under extrusion in 3D printing occurs when the 3D printer doesn't supply enough filament for the print job. This can result in gaps, weak structures, or incomplete layers in the printed object. Hence the objective was to detect this kind of extrusion using images obtained from various 3D printers.

The [dataset](https://www.kaggle.com/competitions/early-detection-of-3d-printing-issues/data) can be viewed here.

So before I could apply any kind of Computer Vision algorithm to this Dataset, some data exploration was needed to understand the constituents of the dataset. 
Here is some of the information that I was able to capture and visualize. 
## EDA
First, we checked the class weightage, which turned out that it wasn't imbalanced. 
![](/images/__results___6_0.png "Pie Chart")

Secondly, we need to check if extrusion is related to a specific kind of printer, i.e. if there is any correlation of the printer with the extrusion caused. As some particular kind of printers might be more susceptible to extrusion.
![](/images/__results___8_1.png "Distribution of images per printer")

## Baseline CNN
Once, we have analysed this data, we can create a simple CNN classifier to check how well it performs. But, there is still one minor issue, all the images are not of the same resolution. So, these images need to be transformed in the pre-processing stage into the same resolution size, before we can process them. Since I'm using PyTorch, I will use Pytorch's image transformation utils to preprocess the image.

```
img_transforms.append(transforms.RandomHorizontalFlip(0.5))
img_transforms.append(transforms.RandomRotation(60))
img_transforms.append(transforms.Resize((400,400), interpolation=transforms.InterpolationMode.NEAREST))    
```

We apply the first 2 transform to make the model invariant of the image orientation and angle. For the final transform, I resized the image to 400x400 size to make the dataset consistent. I experimented with different models like VGG16 and ResNet18 to get some baseline results. Below I'm sharing the training classification report, which indicates a good F1 score, the metric based on which models were compared.

![](/images/__results___9_0.png "Classification report")

## Vision transformers
Based on these results, I was interested in implementing the vision transformer model, as in the papers cited by the competition as reference papers mentioned the use of vision transformers performing better than CNN models. So I used a pertrained vision transformer with a trainable classification layer to perform the classification task.
```
model = ViTForImageClassification.from_pretrained(model_name, num_labels=2, ignore_mismatched_sizes=True)
model.classifier = torch.nn.Linear(in_features=768, out_features=2, bias=True)
model.classifier.requires_grad = True
```

Similar pre-processing steps were used on the training dataset to introduce randomness and size restraints on the data. On performing the training though, I was not able to beat the baseline scores.

The link to my submissions can be viewed here [code](https://github.com/shashvatshah9/3dprinteranomaly/tree/main)


