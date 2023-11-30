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
pip install -r requirements.txt
```
## Data
Download AFHQ dataset and pretrained network.
```bash
bash download.sh afhq-dataset
bash download.sh pretrained-network-afhq
```
Download CelebA-hq dataset and pretrained network.
```bash
bash download.sh celeba-hq-dataset
bash download.sh pretrained-network-celeba-hq
```
