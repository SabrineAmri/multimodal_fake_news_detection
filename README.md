# **Fake News Detection**

This repository contains the code for a project focused on **fake news detection**. The goal of the project is to classify whether news articles are genuine or fake based on multimodal data.

## **Current Features**
- **Data Preprocessing**: Handles text data preprocessing using NLP techniques (tokenization with BERT).
- **Model Training**: Implements a deep learning model based on **BERT** to detect fake news using textual data from the dataset. The training loop includes optimizer setup, loss calculation, and backpropagation.
- **Evaluation**: Model performance is evaluated using accuracy, precision, recall, and F1-score metrics.

## **Project Structure**
multimodal_fake_news_detection \
│ \
├── fake_news_detection.ipynb # Jupyter notebook with the main project code\
│ \
└── README.md # Project documentation \
│\
└── requirements.txt # Dependencies for the project

# **Model: ViLBERT**
This project uses a **BERT-based** classifier for fake news detection based on the text data of news articles. ViLBERT, a model that combines both visual and textual data, is integrated for future development to enable multimodal processing.

**Model Training**
- The model uses BERT from the ```transformers``` library for feature extraction.
- A **binary classifier** is added on top of **BERT** to classify news articles as fake or real.
- The optimizer is set up with **Adam**, and the loss function is **Binary Cross-Entropy** with Logits (**BCEWithLogitsLoss**) for binary classification.
- A training loop is implemented to perform **gradient descent** on the model parameters, using backpropagation to minimize the loss over multiple epochs.

# **Dataset: FakeNewsNet**
The dataset used in this project is [FakeNewsNet dataset](https://github.com/KaiDMML/FakeNewsNet.git), which is a comprehensive dataset for fake news detection. The dataset currently used in the project focuses solely on textual data, including:

- Text: The news article or headline.
- Label: A label indicating whether the news article is fake or real.

FakeNewsNet is built upon two fact-checking websites, **Politifact** and **GossipCop**, and covers a wide variety of topics, making it robust for training fake news detection models. The dataset combines both social and content-based features to improve the reliability of fake news detection.

## **Requirements**
To run the project, you need the following dependencies:
- `torch`
- `transformers`
- `scikit-learn`
- `pandas`
- `numpy`

Since the project dependencies are relatively straightforward, there's no need to include a `requirements.txt` file for now, but you can manually install the necessary dependencies using pip:\
``` pip install torch transformers scikit-learn pandas numpy```

## **Usage**
- Clone the repository using your GitHub token:\
``` git clone https://github.com/SabrineAmri/multimodal_fake_news_detection.git``` 
- Install the dependencies listed above.
- Run the Jupyter notebook file fake_news_detection.ipynb using Jupyter or Google Colab.

## **Results**
The evaluation metrics of the model demonstrate its effectiveness in detecting fake news. The performance is measured using:

- **Accuracy**: The percentage of correctly classified articles.
- **Precision**: The ratio of true positive predictions to the total positive predictions.
- **Recall**: The ratio of true positive predictions to all actual positives.
- **F1 Score**: The harmonic mean of precision and recall.

The current model achieves the following performance on the test dataset:

- **Accuracy**: 97.38%
- **Precision**: 99.27%
- **Recall**: 97.26%
- **F1-Score**: 98.26%

## **Future Work**
- **Multimodality Integration**: Currently, the project "fake_news_detection.ipynb" only handles text data. Future work will involve the integration of multimodal data (both text and image) for enhanced fake news detection.
- **Explainability**: Plan to integrate explainable AI techniques, such as LIME or SHAP, to provide interpretability of the model's predictions.
- **Model Improvements**: Further fine-tuning of the model and experimentation with other transformer-based architectures.

## **Contributing**
Contributions are welcome! Please feel free to submit a pull request or open an issue for discussion.
