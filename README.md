# WOT Official Pytorch implementation
<img src="https://github.com/TienjinHuang/WOT/blob/main/sketch_map.png" width="500" height="300">

**Enhancing Adversarial Training via Reweighting Optimization Trajectory**<br>
Tianjin Huang, Shiwei Liu, Tianlong Chen, Meng Fang, Li Shen, Vlado Menkovski, Lu Yin, Yulong Pei and Mykola Pechenizkiy <br>
https://arxiv.org/abs/2306.14275<br>

Abstract: *Despite the fact that adversarial training has become the de facto method for improving the robustness of deep neural networks, it is well-known that vanilla adversarial training suffers from daunting robust overfitting, resulting in unsatisfactory robust generalization. A number of approaches have been proposed to address these drawbacks such as extra regularization, adversarial weights perturbation, and training with more data over the last few years. However, the robust generalization improvement is yet far from satisfactory. In this paper, we approach this challenge with a brand new perspective -- refining historical optimization trajectories. We propose a new method named \textbf{Weighted Optimization Trajectories (WOT)} that leverages the optimization trajectories of adversarial training in time. We have conducted extensive experiments to demonstrate the effectiveness of WOT under various state-of-the-art adversarial attacks. Our results show that WOT integrates seamlessly with the existing adversarial training methods and consistently overcomes the robust overfitting issue, resulting in better adversarial robustness. For example, WOT boosts the robust accuracy of AT-PGD under AA-L∞ attack by 1.53\% ∼ 6.11\% and meanwhile increases the clean accuracy by 0.55\%∼5.47\% across SVHN, CIFAR-10, CIFAR-100, and Tiny-ImageNet datasets.*

This code base is created by Tianjin Huang [t.huang@tue.nl](mailto:t.huang@tue.nl) during his Ph.D. at Eindhoven University of Technology.<br>

## Usage


