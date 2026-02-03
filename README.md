# Binary Classification of Histopathology Slides using Convolutional Neural Networks


## **Table of Contents**
1. [Introduction](#introduction)
2. [Project Objective](#project-objective)
3. [Dataset Information](#dataset-information)
4. [Data Preprocessing](#data-preprocessing)
5. [User Interface](#user-interface)
6. [Results](#results)
7. [Conclusion](#conclusion)
8. [References](#references)
## **Binary Classification of  Cancer Using Convolutional Neural Networks**
## ⚠️ Disclaimer & License

This project is released under the MIT License – see [LICENSE](LICENSE).

**Disclaimer:**  
This software is provided “as‑is”, without warranty of any kind, express or implied. In no event shall the authors be liable for any claim, damages, or other liability arising from its use.

**Medical use notice:**  
This tool is intended solely for research and educational purposes. It is *not* intended for clinical diagnosis or treatment. Users should not rely on it for medical decisions and should always consult a qualified pathologist or healthcare provider.
## Introduction
Cancer remains one of the most deadly diseases globally, and early diagnosis is often a critical factor in determining patient outcomes. With increasing availability in medical datasets, machine learning has shown immense promise in assisting medical teams in fields such as dermatology [8]. This project investigates binary image classification using a convolution neural network (CNN) to identify the presence of cancer in histopathology slides.


## Project Objective
The primary objective of this project is to develop an efficient and functional convolution neural network (CNN) capable of identifying histopathology images of breast tissue and either cancerous or non-cancerous. Using PyTorch, the model leverages deep learning techniques to analyze patterns in histological images, providing an automated tool for early breast cancer detection. This project is designed for educational and research purposes, making it a valuable resource for students, researchers and anyone interested in exploring medical deep  learning applications.

## Dataset Information
Our images originate from the IDC dataset, available through Kaggle [13], introduced by Janowczyk and Madabhushi [8,9] and consists of over 500,000 labeled PNG histopathology images of breast cells, stored within class-specific folders (“0” = non-cancerous, “1” = cancerous). 



## Data Preprocessing
To ensure optimal model performance, several preprocessing steps were applied to the raw images:

Image Resizing: All images were resized to 224x224 pixels to maintain a consistent input size for the CNN.

Color Normalization: All images were converted to RGB format when necessary to standardize color channels.

Pixel Value Normalization: Pixel intensities were normalized to the range [-1,1] to stabilize and accelerate model convergence.

Train-Test Split: The dataset was divided into 80% for training and 20% for testing using the random_split method to ensure unbiased performance evaluation on unseen data.

## User Interface

To make the model accessible and easy to use, we developed a simple graphical user interface (UI) using Gradio. The UI allows users to upload histopathology images and receive real-time predictions indicating whether the tissue is cancerous or non-cancerous. This is still in progress and it should be known, it should be used for research and educational purposes only. All use of our program follows the license as listed in the LICENSE file.

## Results

Our model was trained on 20 epochs using medical images standardized to 224x224. During this time, our model learned to determine the differences between cancerous and non-canerous images.
During the first few epochs, our training accuracy hovered around 82-90% steadily increasing between each epoch
However by epoch 10, we noticed a slow down in growth reaching an accuracy of 95% while showing very little training loss
Eventually after completing all epochs, we obtained a training accuracy of 96.30% 
Our final test accuracy has shown us that our model generalizes very well to new images.


<!-- Table centered on top -->
<p align="center">
  <img src="https://github.com/user-attachments/assets/eb52943a-67e7-4207-b8da-a63a56a6d77a" width="50%" />
</p>

<p align="center"><b>Table 1: Training Results</b></p>

<!-- Two graphs spaced nicely below -->
<p align="center">
  <img src="https://github.com/user-attachments/assets/73aa6533-f6ba-418e-b591-2ca298084e76" width="40%" />
  &nbsp;&nbsp;&nbsp; <!-- spacing between graphs -->
  <img src="https://github.com/user-attachments/assets/37beb9f5-a61f-4232-a4ed-6df6caee2a2a" width="40%" />
</p>

<p align="center"><b>Figure 2: Training Loss and Accuracy Graphs</b></p>

This model was then saved to be used to train the remaining 20% of images, giving us an overall testing accuracy of 94.73%.

### Discussion

Over the course of 20 epochs, the model has shown to learn quickly and achieved high accuracy on both training and testing sets.

However, although both our test and training show ~95% accuracy, we noticed that our model was both overfitting and underfitting, as shown by the extremely low loss rates, and the subtle difference between training and testing accuracy of ~1.2%. 

We plan to combat this in future iterations by adding data augmentation methods, and experimenting with dropout to further reduce the variance.
## Conclusion
This project demonstrates the potential of deep learning, specifically convolutional neural networks (CNNs), in enhancing model image analysis for breast cancer detection. By leveraging PyTorch and histopathology image data, we developed a model capable of distinguishing between cancerous and non-cancerous breast tissue with a 95.93% accuracy.

## **AI Assistance and Acknowledgment**

Some portions of this project's development, including code snippets, explanations, and debugging support, were assisted using OpenAI's ChatGPT. The tool was used primarily for:

- Clarifying PyTorch syntax and architecture design  
- Suggesting preprocessing and training techniques  
- Improving Markdown formatting and documentation  

All AI-generated content was reviewed, tested, and adapted as necessary to meet the goals of the project.

## References

[1] MLDawn, “How to code a CNN using PyTorch!,” YouTube, https://www.youtube.com/watch?v=aO9naT4IFKI&t=615s

[2] IBM Technology, “What are Convolutional Neural Networks (CNNs)?,” YouTube, https://www.youtube.com/watch?v=QzY57FaENXg

[3] Sentdex, “Convolutional Neural Networks – Deep Learning basics with Python, TensorFlow and Keras p.3,” YouTube, https://www.youtube.com/watch?v=WvoLTXIjBYU&t=815s

[4] CodeBasics, “Dogs vs Cats Image Classification (CNN) | Deep Learning | Python,” YouTube, https://www.youtube.com/watch?v=ENXr1foShrA&t=1431s

[5] T. Reid, “Use PyTorch to Easily Access Your GPU,” Towards Data Science, May 21, 2025, https://towardsdatascience.com/use-pytorch-to-easily-access-your-gpu/

[6] TensorFlow, “Learn TensorFlow,” TensorFlow, https://www.tensorflow.org/learn

[7] H. K. Jeong, C. Park, R. Henao, and M. Kheterpal, “Deep Learning in Dermatology: A Systematic Review of Current Approaches, Outcomes, and Limitations,” *JID Innovations*, vol. 3, no. 1, article 100150, Jan. 2023. [Online]. Available: https://doi.org/10.1016/j.xjidi.2022.100150

[8] J. A. Janowczyk and A. Madabhushi, “Deep learning for digital pathology image analysis: A comprehensive tutorial with selected use cases,” *Journal of Pathology Informatics*, vol. 7, no. 1, p. 29, 2016. doi:10.4103/2153-3539.186902

[9] J. A. Janowczyk and A. Madabhushi, “Automated classification of histopathology images using deep learning,” in *Proc. SPIE 9035, Medical Imaging 2014: Digital Pathology*, 2014. doi:10.1117/12.2043872

[10] G. Litjens et al., “A survey on deep learning in medical image analysis,” *Medical Image Analysis*, vol. 42, Dec 2017. doi:10.1016/j.media.2017.07.005

[11] A. Rakhlin et al., “Deep convolutional neural networks for breast cancer histology image analysis,” *arXiv*, Feb 2018. [Online]. Available: https://arxiv.org/abs/1802.00752

[12] A. Paszke, S. Gross, F. Massa, et al., “PyTorch: An Imperative Style, High-Performance Deep Learning Library,” in *Advances in Neural Information Processing Systems 32 (NeurIPS 2019)*, 2019, pp. 8024–8035.


Special Thanks to Professor Robertazzi, for the advice he gave us during this process
