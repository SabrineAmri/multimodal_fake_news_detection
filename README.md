# **Multimodal Fake News Detection**

This repository contains the code for a project focused on **multimodal fake news detection**. The goal of the project is to classify whether news articles, leveraging different modalities (such as text), are genuine or fake.

## **Current Features**
- **Data Preprocessing**: Handles text data preprocessing using NLP techniques.
- **Model Training**: Implements a deep learning model to detect fake news using textual data.
- **Evaluation**: Model performance is evaluated using accuracy, precision, recall, and F1-score metrics.

## **Project Structure**
multimodal_fake_news_detection/ │ ├── multimodal_fake_news_detection.ipynb # Jupyter notebook with the main project code ├── sample_data/ # Folder for any sample datasets └── README.md # Project documentation

## **Requirements**
To run the project, you need the following dependencies:
- `torch`
- `transformers`
- `scikit-learn`
- `pandas`
- `numpy`

Since the project dependencies are relatively straightforward, there's no need to include a `requirements.txt` file for now, but you can manually install the necessary dependencies using pip:
pip install torch transformers scikit-learn pandas numpy

## **Usage**
Clone the repository:
git clone https://github.com/SabrineAmri/multimodal_fake_news_detection.git
Install the dependencies listed above.
Run the Jupyter notebook file multimodal_fake_news_detection.ipynb using Jupyter or Google Colab.

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
- **Explainability**: Plan to integrate explainable AI techniques, such as LIME or SHAP, to provide interpretability of the model's predictions.
- **Model Improvements**: Further fine-tuning of the model and experimentation with other transformer-based architectures.

## **Contributing**
Contributions are welcome! Please feel free to submit a pull request or open an issue for discussion.



