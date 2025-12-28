<h2>TensorFlow-FlexUNet-Image-Segmentation-HRSID-Satellite-Imagery-Ship  (2025/12/29)</h2>
Toshiyuki Arai<br>
Software Laboratory antillia.com<br><br>
This is the first experiment of Image Segmentation for <b>High resolution sar images dataset (HRSID) for Ship Detection </b> based on our <a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet</a> (TensorFlow Flexible UNet Image Segmentation Model for Multiclass) , 
and 
<a href="https://drive.google.com/open?id=1BZTU8Gyg20wqHXtBPFzRazn_lEdvhsbE">HRSID_JPG.rar </a> on Google drive 
in <a href="https://github.com/chaozhong2010/HRSID">HRSID</a>
<br><br>
<hr>
<b>Actual Image Segmentation for MARS HRSID Images of 800x800 pixels</b><br>
As shown below, the inferred masks predicted by our segmentation model trained by the dataset appear similar to the ground truth masks, but they lack precision in certain areas.
<br><br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/HRSID/mini_test/images/P0001_600_1400_9600_10400.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HRSID/mini_test/masks/P0001_600_1400_9600_10400.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HRSID/mini_test_output/P0001_600_1400_9600_10400.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/HRSID/mini_test/images/P0015_3600_4400_4200_5000.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HRSID/mini_test/masks/P0015_3600_4400_4200_5000.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HRSID/mini_test_output/P0015_3600_4400_4200_5000.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/HRSID/mini_test/images/P0008_0_800_9000_9800.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HRSID/mini_test/masks/P0008_0_800_9000_9800.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HRSID/mini_test_output/P0008_0_800_9000_9800.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>1  Dataset Citation</h3>
The dataset used here was derived from <br><br>
<a href="https://drive.google.com/open?id=1BZTU8Gyg20wqHXtBPFzRazn_lEdvhsbE">HRSID_JPG.rar </a> on Google drive
in <a href="https://github.com/chaozhong2010/HRSID">HRSID</a>

<br><br>
<b>HRSID</b><br>
High resolution sar images dataset (HRSID) is a data set for ship detection, semantic segmentation, and instance 
segmentation tasks in high-resolution SAR images. <br>
This dataset contains a total of 5604 high-resolution SAR images and 16951 ship instances. HRSID dataset draws on the construction process of 
the Microsoft Common Objects in Context (COCO) datasets, including SAR images with different resolutions, polarizations, 
sea conditions, sea areas, and coastal ports.<br> 
This dataset is a benchmark for researchers to evaluate their approaches. For HRSID, the resolution of SAR images is as follows: 0.5m, 1 m, and 3 m.
<br>
<br>
<b>Citation</b><br>
[1] Shunjun Wei ; Xiangfeng Zeng ; Qizhe Qu ; Mou Wang ; Hao Su ; Jun Shi. <br>
<a href="https://ieeexplore.ieee.org/document/9127939?denied=">
<b>HRSID: A High-Resolution SAR Images Dataset for Ship Detection and Instance Segmentation . IEEE Access</b>
</a>
<br>
<br>
<b>License</b><br>
<a href="https://github.com/chaozhong2010/HRSID?tab=GPL-3.0-1-ov-file#readme"> GPL-3.0 license</a>
<br>
<br>
<h3>
2 HRSID ImageMask Dataset
</h3>
 If you would like to train this HRSID Ship Segmentation model by yourself,
please down load the <a href="https://drive.google.com/open?id=1BZTU8Gyg20wqHXtBPFzRazn_lEdvhsbE">HRSID_JPG.rar </a> on Google drive
, and expand the downloaded in a working directory.<br>
The folder structure of the <b>HRSID_JPG</b> dataset is the following.<br> 
<pre>
./HRSID_JPG
├─annotations
│   ├─test2017.json
│   ├─train_test2017.json
│   └─train2017.json
└─JPEGImages
      ├─P0001_0_800_7200_8000.jpg
      ...
      └─P0137_109
</pre>
We used a Python script <a href="./generator/ImageMaskDatasetGenerator.py">ImageMaskDatasetGenerator.py</a> to generate 
our train HRSID-Image-Mask subset from the JSON file <b>annotations/train2017.json</b> and the JPG files in <b>JPEGImages</b>. and split the
Image-Mask dataset into <b>test</b> , <b>train</b> and <b>valid </b> subsets.<br>
<pre>
./dataset
└─HRSID
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<br>
<b>HRSID Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/HRSID/HRSID_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is large enough to use for a training set of our segmentation model.
<br><br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/HRSID/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/HRSID/asset/train_masks_sample.png" width="1024" height="auto">
<br>

<h3>
3 Train TensorflowFlexUNet Model
</h3>
 We trained HRSID TensorflowFlexUNet Model by using the following
<a href="./projects/TensorFlowFlexUNet/HRSID/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/HRSID and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters=16</b> and a large <b>base_kernels=(9,9)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorflowUNet.py">TensorflowUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
image_width    = 512
image_height   = 512
image_channels = 3
input_normalize = True
normalization  = False

num_classes    = 2

base_filters   = 16
base_kernels  = (9,9)
num_layers    = 8

dropout_rate   = 0.05
dilation       = (1,1)
</pre>

<b>Learning rate</b><br>
Defined a small learning rate.  
<pre>
[model]
learning_rate  =  0.0001
</pre>

<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and "dice_coef_multiclass".<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b >Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.4
reducer_patience   = 4
</pre>
<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>
<b></b><br>
<b>RGB color map</b><br>
rgb color map dict for HRSID 1+1 classes.<br>
<pre>
[mask]
mask_file_format = ".png"
;HRSID 1+1
rgb_map = {(0,0,0):0,  (0, 255, 0):1,  }       
</pre>
<b>Epoch change inference callbacks</b><br>
Enabled epoch_change_infer callback.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
epoch_changeinfer        = False
epoch_changeinfer_dir    = "./epoch_changeinfer"
num_infer_images         = 6
</pre>
By using this epoch_change_infer callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 
<b>Epoch_change_inference output at starting (1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/HRSID/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (19,20,21)</b><br>
<img src="./projects/TensorFlowFlexUNet/HRSID/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at ending (39,40,41)</b><br>
<img src="./projects/TensorFlowFlexUNet/HRSID/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>

<br>
In this experiment, the training process was stopped at epoch 41 by EarlyStoppingCallback.<br><br>
<img src="./projects/TensorFlowFlexUNet/HRSID/asset/train_console_output_at_epoch41.png" width="880" height="auto"><br>
<br>
<a href="./projects/TensorFlowFlexUNet/HRSID/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/HRSID/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/HRSID/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/HRSID/eval/train_losses.png" width="520" height="auto"><br>
<br>
<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/HRSID</b> folder,<br>
and run the following bat file to evaluate TensorflowFlexUNet model for HRSID.<br>
<pre>
>./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNetEvaluator.py  ./train_eval_infer.config
</pre>
Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/HRSID/asset/evaluate_console_output_at_epoch41.png" width="880" height="auto">
<br><br>Image-Segmentation-HRSID

<a href="./projects/TensorFlowFlexUNet/HRSID/evaluation.csv">evaluation.csv</a><br>
The loss (categorical_crossentropy) to this HRSID/test was low, and dice_coef_multiclass high as shown below.
<br>
<pre>
categorical_crossentropy,0.0064
dice_coef_multiclass,0.9973
</pre>
<br>
<h3>5 Inference</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/HRSID</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorflowFlexUNet model for HRSID.<br>
<pre>
>./3.infer.bat
</pre>
This simply runs the following command.
<pre>
>python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/HRSID/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/HRSID/asset/mini_test_masks.png" width="1024" height="auto"><br>
<hr>
<b>Inferred test masks</b><br>
<img src="./projects/TensorFlowFlexUNet/HRSID/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks for HRSID Images of 800x800 pixels</b><br>
As shown below, the inferred masks predicted by our segmentation model trained by the dataset appear similar to the ground truth masks, but they lack precision in certain areas.
<br>
<br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/HRSID/mini_test/images/P0001_600_1400_9600_10400.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HRSID/mini_test/masks/P0001_600_1400_9600_10400.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HRSID/mini_test_output/P0001_600_1400_9600_10400.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/HRSID/mini_test/images/P0001_4200_5000_600_1400.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HRSID/mini_test/masks/P0001_4200_5000_600_1400.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HRSID/mini_test_output/P0001_4200_5000_600_1400.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/HRSID/mini_test/images/P0008_0_800_9000_9800.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HRSID/mini_test/masks/P0008_0_800_9000_9800.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HRSID/mini_test_output/P0008_0_800_9000_9800.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/HRSID/mini_test/images/P0015_3600_4400_4200_5000.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HRSID/mini_test/masks/P0015_3600_4400_4200_5000.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HRSID/mini_test_output/P0015_3600_4400_4200_5000.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/HRSID/mini_test/images/P0017_600_1400_7800_8600.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HRSID/mini_test/masks/P0017_600_1400_7800_8600.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HRSID/mini_test_output/P0017_600_1400_7800_8600.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/HRSID/mini_test/images/P0021_0_800_7800_8600.jpg" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HRSID/mini_test/masks/P0021_0_800_7800_8600.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/HRSID/mini_test_output/P0021_0_800_7800_8600.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>
References
</h3>
<b>1. HRSID: A High-Resolution SAR Images Dataset for Ship Detection and Instance Segmentation</b><br>
 Shunjun Wei ; Xiangfeng Zeng ; Qizhe Qu ; Mou Wang ; Hao Su ; Jun Shi. <br>
<a href="https://ieeexplore.ieee.org/document/9127939?denied=">
https://ieeexplore.ieee.org/document/9127939?denied=
</a>
<br>
<br>
<b>2. Instance segmentation ship detection based on improved Yolov7 using complex background SAR images</b><br>
Muhammad Yasir, Lili Zhan, Shanwei Liu, Jianhua Wan, Md Sakaouth Hossain,<br>
Arife Tugsan Isiacik Colak, Mengge Liu, Qamar Ul Islam, Syed Raza Mehdi, Qian Yang<br>
<a href="https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2023.1113669/full">
https://www.frontiersin.org/journals/marine-science/articles/10.3389/fmars.2023.1113669/full</a>
<br><br>

<b>3. TensorFlow-FlexUNet-Image-Segmentation-Model</b><br>
Toshiyuki Arai <br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model
</a>
<br>
<br>
