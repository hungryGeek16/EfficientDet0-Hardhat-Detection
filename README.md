# 1. EfficientDet and it's Structure:

*  This application of object detection has three main stages:

      a.   **EfficientNet**  
      b.   **BiFPN(Bi-directional feature pyramid network)**  
      c.   **Bounding Box/Classification Heads** 

## a. EfficientNet:

* Backbone architectures are very important for an object detection module, whose job is to make features more significant for detection. 
* There are various approches by which these backbones attain efficiency in producing features. Out which is incresing width(feature channels), depth(adding more layers) and resolution, is common practice.

**Fig: Effect of increasing width, depth and resolution.**

* So efficientnet focuses on increasing these parameters with a constant ratio. [Refer] it's paper to know ow it's done.

* It's basic structure is given below


* **MB Convs:** Input -> 1 x 1 Conv op -> 3 x 3 Depthwise Conv op -> 1 x 1 Conv + Input



## b. BiFPN:

* When we look at FPN network,s every node has one outward connection with every other node in the network. 

* Hence every node would have less contribution to the performance of the network. 

* To gain over this drawback, the paper introduces bi-direction feature pyramid network to get more accuracy. [Refer] it's paper to know more.


## c. Classification and Detection Head:

* After FPN layer, output features are passsed to 2 different heads deployed for classification and detection which give out confidence score and bbox offsets. 


# 2. Why choose efficientdet ?

* The idea is simple over here. When accuracy is the only goal, then deeper and heavier networks like Faster-RCNN would perform with precise accuracy and recall.

* On the other hand, if the goal is speed, then MobilenetV2-ssd , YOLO-tiny variants will do the job. But if it's both, then EfficientDet is preferrable.

* To play safe with not much of a tradeoff between accuracy and speed, EfficientDet was chosen.




# 3. Training and Inferenece:

* Refer the colab over here to train the model

## Inference

* Clone this repository.

```bash
cd $ROOT/EfficientDet0-Hardhat-Detection
!pip install -r requirements.txt
!unzip models.zip
!rm models.zip
```
* Download fine_tuned_model folder from [here]().

* Inference with this command:
```bash
python inference.py /path/to/image /path/to/model_folder
```
* Inference video:
```bash
python infer_vid.py /path/to/image /path/to/model_folder
```

# 4. Task which are complete for now:
