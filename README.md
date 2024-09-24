Multimodal Fake News Detection
This repository contains the code for a project focused on multimodal fake news detection. The goal of the project is to classify whether news articles, leveraging different modalities (such as text), are genuine or fake.

Current Features
Data Preprocessing: Handles text data preprocessing using NLP techniques, including tokenization and vectorization.
Model Training: Implements a deep learning model to detect fake news using textual data, with the possibility of extending to other modalities like images or video in future versions.
Evaluation: Model performance is evaluated using accuracy, precision, recall, and F1-score metrics.
Project Structure
multimodal_fake_news_detection.ipynb: Jupyter notebook containing the implementation of the fake news detection model, data preprocessing, training, and evaluation.
Requirements
To run the project, you need the following dependencies:

torch
transformers
scikit-learn
pandas
numpy
You can install all dependencies using:

bash
Copy code
pip install -r requirements.txt
Usage
To run the model, follow these steps:

Clone the repository to your local machine:

bash
Copy code
git clone https://github.com/SabrineAmri/multimodal_fake_news_detection.git
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Open the multimodal_fake_news_detection.ipynb notebook and follow the steps inside to train the model and evaluate it on your dataset.

The notebook will print out evaluation metrics such as accuracy, precision, recall, and F1-score after training.

Results
The current model achieves the following performance on the test dataset:

Accuracy: 97.38%
Precision: 99.27%
Recall: 97.26%
F1-Score: 98.26%
Future Work
Explainability: Plan to integrate explainable AI techniques, such as LIME or SHAP, to provide interpretability of the model's predictions.
Multimodal Extension: Future versions will explore additional modalities, such as images and videos, to enhance the detection of fake news.
Model Improvements: Further fine-tuning of the model and experimentation with other transformer-based architectures.
Contributing
Contributions are welcome! If you have any ideas or improvements, feel free to fork the repository, create a new branch, and submit a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for more details.