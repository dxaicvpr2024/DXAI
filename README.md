# DXAI: Explaining Classification by Image Decomposition
## Abstract
We propose a new way to explain and to visualize neural network classification through a decomposition-based explainable AI (DXAI).
Instead of providing an explanation heatmap, our method yields a decomposition of the image into class-agnostic and class-distinct parts, with respect to the data and chosen classifier. Following a fundamental signal processing paradigm of analysis and synthesis, the original image is the sum of the decomposed parts. We thus obtain a radically different way of explaining classification. The class-agnostic part ideally is composed of all image features which do not posses  class information, where the class-distinct part is its complementary.
This new perceptual visualization, can be more helpful and informative in certain scenarios, especially when the attributes are dense, global and additive in nature, for instance, when colors or textures are essential for class distinction.
![Heatmaps compare](https://github.com/dxaicvpr2024/DXAI/blob/main/heatmaps_compare.jpg)

## Installation
The code in this repository is written based on the stragan-v2 code that can be found [here](https://github.com/clovaai/stargan-v2).
Therefore the installation of the packages and the requirements are similar. To install:

```bash
git clone https://github.com/dxaicvpr2024/DXAI.git
cd DXAI/
conda create -n dxai python=3.6.7
conda activate dxai
conda install -y pytorch=1.4.0 cudatoolkit=10.0 -c pytorch
pip install torchvision
conda install x264=='1!152.20180717' ffmpeg=4.0.2 -c conda-forge
pip install opencv-python==4.1.2.30 ffmpeg-python==0.2.0 scikit-image==0.16.2
pip install pillow==7.0.0 scipy==1.2.1 tqdm==4.43.0 munch==2.5.0
pip install opencv-python
pip install captum
```
## Data

