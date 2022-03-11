# The Re-implementations of Shallow Domain Adaptation Methods

This repository integrates the codes for some fantastic shallow domain adaptation methods. Some were provided by the authors (from their original papers), and the others were re-implemented  by me. They are generally operational, but their hyper-parameters still need to be tuned. Besides,  **the following codes has not been rigorously tested**, and may contain errors. 

If you find any errors or need any help in reproducing the code for shallow domain adaptation, please feel free to contact me. 

If any author or publisher has questions, please contact me to remove or replace them.

my email is coding495@163.com.

---

'Demo.m' gives a simple examples. 

To run the codes, the size of the inputs are: <img src="http://latex.codecogs.com/svg.latex?X_s\in&space;\mathbb{R}^{m\times&space;n_s},X_t\in&space;\mathbb{R}^{m\times&space;n_t},Y_s\in&space;\mathbb{R}^{&space;n_s},Y_t\in&space;\mathbb{R}^{m\times&space;n_t}" title="http://latex.codecogs.com/svg.latex?X_s\in \mathbb{R}^{m\times n_s},X_t\in \mathbb{R}^{m\times n_t},Y_s\in \mathbb{R}^{ n_s},Y_t\in \mathbb{R}^{m\times n_t}" />, where *m* is the dimension, and <img src="http://latex.codecogs.com/svg.latex?n_s" title="http://latex.codecogs.com/svg.latex?n_s" /> and <img src="http://latex.codecogs.com/svg.latex?n_t" title="http://latex.codecogs.com/svg.latex?n_t" /> present the number of the source and target samples, respectively (<img src="http://latex.codecogs.com/svg.latex?Y_t" title="http://latex.codecogs.com/svg.latex?Y_t" /> is used to calculate the accuracy only, and is not involved in training).

---

- 2015-LRSR [[1]](https://ieeexplore.ieee.org/abstract/document/7360924)
- 2017-VDA [[2]](https://link.springer.com/article/10.1007/s10115-016-0944-x)
- 2017-JGSA [[3]](https://ieeexplore.ieee.org/document/8100030)
- 2018-DICD [[4]](https://ieeexplore.ieee.org/abstract/document/8362753/)
- 2018-TLR [[5]](https://ieeexplore.ieee.org/abstract/document/8486513)
- 2019-DICE [[6]](https://ieeexplore.ieee.org/abstract/document/8353356):  [provided by authors](https://liangjian.xyz/code/uda_code.rar)
- 2019-DISA [[7]](https://link.springer.com/article/10.1007/s42044-019-00037-y):  [provided by authors](https://github.com/jtahmores/DISA)
- 2019-SPDA [[8]](https://www.sciencedirect.com/science/article/pii/S0925231219300979)
- 2020-ATL [[9]](https://ieeexplore.ieee.org/abstract/document/8649674)
- 2020-DAC [[10]](https://www.sciencedirect.com/science/article/pii/S0950705119306082)
- 2020-DGA [[11]](https://ieeexplore.ieee.org/abstract/document/8961922/)
- 2020-DGB-DA [[21]](https://ieeexplore.ieee.org/abstract/document/9534057/)
- 2020-SPL [[16](https://ojs.aaai.org/index.php/AAAI/article/view/6091)]: [provided by authors](https://github.com/hellowangqian/domain-adaptation-capls)
- 2020-DGSA [[12]](https://ieeexplore.ieee.org/abstract/document/9115265)
- 2020-DSL-DGDA [[13]](https://link.springer.com/article/10.1007/s10489-019-01610-5) 
- 2020-McDA [[18]](https://link.springer.com/article/10.1007/s11063-019-10090-0)
- 2020-WCS-RAR [[17]](https://www.sciencedirect.com/science/article/pii/S0893608020300113)
- 2021-JDSC [[14]](https://link.springer.com/article/10.1007/s11760-020-01745-w):  [provided by authors](https://github.com/jtahmores/JDSC)
- 2021-PDALC [[15]](https://ieeexplore.ieee.org/abstract/document/9428235)
- 2021-CMFC [[20]](https://www.sciencedirect.com/science/article/pii/S0045790621000604)
- 2021-CDEM [[19]](https://link.springer.com/chapter/10.1007/978-3-030-73197-7_29) :  [provided by authors](https://github.com/yuntaodu/CDEM)

# Reference

[1] Y. Xu, X. Fang, J. Wu, X. Li and D. Zhang, "Discriminative Transfer Subspace Learning via Low-Rank and Sparse Representation," in IEEE Transactions on Image Processing, vol. 25, no. 2, pp. 850-863, Feb. 2016, doi: 10.1109/TIP.2015.2510498.

[2] Tahmoresnezhad, J., Hashemi, S. Visual domain adaptation via transfer feature learning. *Knowl Inf Syst* 50, 585–605 (2017). https://doi.org/10.1007/s10115-016-0944-x

[3] J. Zhang, W. Li and P. Ogunbona, "Joint Geometrical and Statistical Alignment for Visual Domain Adaptation," 2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2017, pp. 5150-5158, doi: 10.1109/CVPR.2017.547.

[4] S. Li, S. Song, G. Huang, Z. Ding and C. Wu, "Domain Invariant and Class Discriminative Feature Learning for Visual Domain Adaptation," in IEEE Transactions on Image Processing, vol. 27, no. 9, pp. 4260-4273, Sept. 2018, doi: 10.1109/TIP.2018.2839528.

[5] P. Xiao, B. Du, J. Wu, L. Zhang, R. Hu and X. Li, "TLR: Transfer Latent Representation for Unsupervised Domain Adaptation," *2018 IEEE International Conference on Multimedia and Expo (ICME)*, 2018, pp. 1-6, doi: 10.1109/ICME.2018.8486513.

[6] J. Liang, R. He, Z. Sun and T. Tan, "Aggregating Randomized Clustering-Promoting Invariant Projections for Domain Adaptation," in IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 41, no. 5, pp. 1027-1042, 1 May 2019, doi: 10.1109/TPAMI.2018.2832198.

[7] Rezaei, S., Tahmoresnezhad, J. Discriminative and domain invariant subspace alignment for visual tasks. *Iran J Comput Sci* 2, 219–230 (2019). https://doi.org/10.1007/s42044-019-00037-y.

[8] Xiao, Ting, et al. "Structure preservation and distribution alignment in discriminative transfer subspace learning." *Neurocomputing* 337 (2019): 218-234.

[9] Z. Peng, W. Zhang, N. Han, X. Fang, P. Kang and L. Teng, "Active Transfer Learning," in IEEE Transactions on Circuits and Systems for Video Technology, vol. 30, no. 4, pp. 1022-1036, April 2020, doi: 10.1109/TCSVT.2019.2900467.

[10] Wang, Yunyun, et al. "Soft large margin clustering for unsupervised domain adaptation." *Knowledge-Based Systems* 192 (2020): 105344, https://doi.org/10.1016/j.knosys.2019.105344.

[11] L. Luo, L. Chen, S. Hu, Y. Lu and X. Wang, "Discriminative and Geometry-Aware Unsupervised Domain Adaptation," in IEEE Transactions on Cybernetics, vol. 50, no. 9, pp. 3914-3927, Sept. 2020, doi: 10.1109/TCYB.2019.2962000.

[12] J. Zhao, L. Li, F. Deng, H. He and J. Chen, "Discriminant Geometrical and Statistical Alignment With Density Peaks for Domain Adaptation," in IEEE Transactions on Cybernetics, vol. 52, no. 2, pp. 1193-1206, Feb. 2022, doi: 10.1109/TCYB.2020.2994875.

[13] Gholenji, E., Tahmoresnezhad, J. Joint discriminative subspace and distribution adaptation for unsupervised domain adaptation. *Appl Intell* 50, 2050–2066 (2020). https://doi.org/10.1007/s10489-019-01610-5

[14] Noori Saray, S., Tahmoresnezhad, J. Joint distinct subspace learning and unsupervised transfer classification for visual domain adaptation. *SIViP* 15, 279–287 (2021). https://doi.org/10.1007/s11760-020-01745-w

[15] Y. Li, D. Li, Y. Lu, C. Gao, W. Wang and J. Lu, "Progressive Distribution Alignment Based on Label Correction for Unsupervised Domain Adaptation," 2021 IEEE International Conference on Multimedia and Expo (ICME), 2021, pp. 1-6, doi: 10.1109/ICME51207.2021.9428235.

[16] Wang, Q., & Breckon, T. (2020). Unsupervised Domain Adaptation via Structured Prediction Based Selective Pseudo-Labeling. *Proceedings of the AAAI Conference on Artificial Intelligence*, *34*(04), 6243-6250. https://doi.org/10.1609/aaai.v34i04.6091

[17] Yang, Liran, and Ping Zhong. "Robust adaptation regularization based on within-class scatter for domain adaptation." *Neural Networks* 124 (2020): 60-74.

[18] Zhang, W., Zhang, X., Lan, L. *et al.* Maximum Mean and Covariance Discrepancy for Unsupervised Domain Adaptation. *Neural Process Lett* 51, 347–366 (2020). https://doi.org/10.1007/s11063-019-10090-0

[19] Du, Yuntao, et al. "Cross-domain error minimization for unsupervised domain adaptation." *International Conference on Database Systems for Advanced Applications*. Springer, Cham, 2021.

[20] Chang H, Zhang F, Ma S, et al. Unsupervised domain adaptation based on cluster matching and Fisher criterion for image classification[J]. Computers & Electrical Engineering, 2021, 91: 107041.

[21] Y. Du, D. Zhou, J. Shi, Y. Lei and M. Gong, "Dynamic-graph-based Unsupervised Domain Adaptation," 2021 International Joint Conference on Neural Networks (IJCNN), 2021, pp. 1-7, doi: 10.1109/IJCNN52387.2021.9534057.