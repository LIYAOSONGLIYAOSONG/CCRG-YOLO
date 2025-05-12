![github1](https://github.com/user-attachments/assets/ba37b997-516a-4a22-9f8d-572c05aa7cf6)

Based on the YOLO model, this study proposes an improved model called CCRG-YOLO, which introduces several key enhancements over the original. First, the multi-head convolution context enhancement feature refinement algorithm is incorporated, allowing the network to accurately identify small targets of varying sizes. Additionally, to enhance the model's expressiveness, generalization, and adaptability, the detection head utilizes the content-aware reorganization reparameterized generalized feature pyramid network (CRepGFPN), replacing the original feature pyramid network. This new network structure effectively combines high-level semantic information with low-level spatial information, facilitating multi-scale feature fusion between different layers. Cross-layer connections are also employed, enabling more effective information transmission and allowing the network to expand deeper, thereby aggregating context information over a larger receptive field and addressing the limitations of the original model's receptive field.
Moreover, spatial perception capabilities are introduced to the convolutional layers in the pyramid network's output, improving the prediction model's ability to locate objects and enhancing the overall detection performance. The implementation of the ShuffleAttention mechanism further reduces computational complexity, boosts generalization, and enhances feature representation. As a result, the proposed CCRG-YOLO model demonstrates strong adaptability and a lightweight design, significantly improving the performance and efficiency of deep learning models.
In summary, the main contributions of this study in steel surface defect detection are as follows:
1. Propose a CCRG-YOLO network based on YOLOv5 design for steel surface defect detection. Based on a large amount of relevant information, this paper is the first to propose a content-aware reorganization and re-parameterization generalized feature pyramid network, which has a better feature fusion strategy and enhances the network's expression and generalization capabilities.
2. The new structure of this study can not only aggregate contextual information within a large receptive field, avoid the insufficient receptive field caused by the nearest neighbor and bilinear interpolation, but also change the fixed kernel to adapt to instance-specific content in a timely manner.
3. A multi-head convolution context-enhanced feature refinement algorithm is proposed, which has the advantages of high parameter deployment efficiency and strong global information capture capability. The introduced channel attention mechanism and spatial feature refinement suppress the small targets that appear in the multi-scale feature fusion process through dilated convolution and disappear in the conflicting information.
4. The proposed pyramid network output is endowed with spatial perception ability, and two channels are added after the feature map, so that the network can learn translation invariance and dependency in different scene requirements.
![github2](https://github.com/user-attachments/assets/4cc4a80f-09a9-46ac-91ce-1f2a8d3fad0c)
![github3](https://github.com/user-attachments/assets/528f5bb3-f990-40ca-9753-61108292638e)

**
[  
Citation  
@article{
  title={Enhanced Steel Surface Defect Detection via Multiscale Small Object Detection with CCRG-YOLO Model},  
  author={yangjiasheng and liyaosong, yuanjianglong and xiachenbo, ningzihao and zhangzaihong},  
  journal={The Visual Computer},  
  year={2025},  
  publisher={Springer Berlin Heidelberg}  
}  
]  
**  


