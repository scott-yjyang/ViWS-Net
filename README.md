# [ICCV 2023] Video Adverse-Weather-Component Suppression Network via Weather Messenger and Adversarial Backpropagation

This project proposes the first method for **video adverse weather removal** task.

[paper](https://openaccess.thecvf.com/content/ICCV2023/html/Yang_Video_Adverse-Weather-Component_Suppression_Network_via_Weather_Messenger_and_Adversarial_Backpropagation_ICCV_2023_paper.html).

[project website](https://yijun-yang.github.io/viwsnet/index.html).


## News
- 23-11-12. This paper has been accepted by ICCV 2023. Code is still quickly updating 🌝.

### Abstract

Although convolutional neural networks (CNNs) have been proposed to remove adverse weather conditions in single images using a single set of pre-trained weights, they fail to restore weather videos due to the absence of temporal information. Furthermore, existing methods for removing adverse weather conditions (e.g., rain, fog, and snow) from videos can only handle one type of adverse weather. In this work, we propose the first framework for restoring videos from all adverse weather conditions by developing a video adverse-weather-component suppression network (ViWS-Net).
To achieve this, we first devise a weather-agnostic video transformer encoder with multiple transformer stages. Moreover, we design a long short-term temporal modeling mechanism for weather messenger to early fuse input adjacent video frames and learn weather-specific information. We further introduce a weather discriminator with gradient reversion, to maintain the weather-invariant common information and suppress the weather-specific information in pixel features, by adversarially predicting weather types. Finally, we develop a messenger-driven video transformer decoder to retrieve the residual weather-specific feature, which is spatiotemporally aggregated with hierarchical pixel features and refined to predict the clean target frame of input videos.
Experimental results, on benchmark datasets and real-world weather videos, demonstrate that our ViWS-Net outperforms current state-of-the-art methods in terms of restoring videos degraded by any weather condition.


### Requirements

  ```
  conda env create -f environment.yaml
  ```

### Dataset

RainMotion [[dataset](https://drive.google.com/file/d/1905B_e2RgQGnyfHd5xpjB4lTLYoq0Jm4/view?usp=sharing)]
REVIDE [[dataset](https://drive.google.com/file/d/1MYaVMUtcfqXeZpnbsfoJ2JBcpZUUlXGg/view?usp=sharing)]
KITTI-snow [[dataset](https://drive.google.com/file/d/1_1IsyT5nTvYjrCwNgP4LDOh_rXnPP_LE/view?usp=sharing)]

### Model
[Pretrained-weights] (https://drive.google.com/drive/folders/15iZKXFT7apjUSoN2WUMAbb0tvJgyh3YP)
[Best_Checkpoint](https://drive.google.com/file/d/1Jfui4eaDY24CPRsQQqBjlA9bUhE6G76b/view?usp=drive_link)

### Setup

- Training

  ```
  python main_multi.py --batchSize 4 --data_dir Dataset/RainMotion/Test --save_folder weights
  ```
- Testing
- ```
  python eval_derain.py --data_dir Dataset/RainMotion/Test --model weights/model_motion.pth --output Results
  python eval_psnr_ssim.py --dataset RainMotion
  ```


## Cite
If you find this code useful, please cite
~~~
@inproceedings{yang2023video,
  title={Video Adverse-Weather-Component Suppression Network via Weather Messenger and Adversarial Backpropagation},
  author={Yang, Yijun and Aviles-Rivero, Angelica I and Fu, Huazhu and Liu, Ye and Wang, Weiming and Zhu, Lei},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={13200--13210},
  year={2023}
}
~~~

