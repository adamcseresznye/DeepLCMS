DeepLCMS: A framework that leverages transfer learning for the classification of pseudo images in mass spectrometry-based analysis
==============================

![Alt Text](demo.png "Examples of DeepLCMS predictions with probability estimates.")

DeepLCMS is an open-source project, that aims to provide researchers with a reproducible code-template for leveraging deep learning for mass spectrometry data analysis.

In contrast to traditional methods that involve laborious steps like peak alignment, data annotation, and quantitation, DeepLCMS delivers results significantly faster by approaching LC/MS problems as a **computer vision task**. This novel approach uses a neuronal network to directly learn from the patterns inherent in the sample in an unbiased way without the need for manual intervention, accelerating the entire workflow.

The study stands out from previous research by conducting a comprehensive evaluation of diverse architecture families, including cutting-edge architectures like vision transformers. Additionally, it employs basic hyperparameter tuning to optimize key parameters such as the optimizer and learning rate scheduler. Furthermore, it examines the impact of image pretreatment on validation metrics, exploring image sharpness and data augmentation techniques that mimic retention time shift. To enhance model generalization, this study takes advantage of regularization techniques like random-tilting images and random erasing during training. Finally, it also explores model interpretability by delving into the decision-making process of the pre-trained network, employing TorchVision for a comprehensive analysis.

For more information visit [my website](https://adamcseresznye.github.io/projects/DeepLCMS/DeepLCMS.html).

Quick Overview of Project Organization
------------

    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── deeplcms_functions  <- module to inspect metabolomicsworkbench database, convert and preprocess images (this module is mainly used with CPU)
    │   ├── train_google_colab       <- module to train CNNs with corresponding notebooks. This folder can be uploaded, in zipped format, to Google Colab to take advantage the free GPU
