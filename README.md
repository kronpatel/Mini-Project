MINI PROJECT: MENTAL STRESS DETECTION IN STUDENTS USING MACHINE LEARNING

Project Overview:
-----------------
This project aims to detect mental stress in students by analyzing their text statements using machine learning and natural language processing (NLP) techniques. The model classifies input text as either "Stress" or "No Stress".

Features:
---------
- Cleans and preprocesses student text data
- Visualizes common words using a word cloud
- Trains a Naive Bayes classifier for stress detection
- Predicts stress level from user input in real-time

Project Structure:
------------------
Mini-Project/
│
├── stress_detection.py      # Main Python script
├── stress.csv              # Dataset (text, label)
├── requirements.txt        # Python dependencies
└── README.txt              # Project documentation

Requirements:
-------------
- Python 3.7+
- pandas
- numpy
- nltk
- matplotlib
- wordcloud
- scikit-learn

Install all dependencies with:
pip install -r requirements.txt

Dataset Format:
---------------
The dataset (stress.csv) should have the following columns:

text                                               | label
---------------------------------------------------|------
I am feeling very stressed about my exams.         | 1
I am happy and relaxed today.                      | 0
Assignments are piling up and I can't sleep.       | 1
I enjoyed my day with friends.                     | 0

label: 1 = Stress, 0 = No Stress

How to Run:
-----------
1. Clone the repository
   git clone https://github.com/kronpatel/Mini-Project
   cd Mini-Project

2. Install dependencies
   pip install -r requirements.txt

3. Run the script
   python stress_detection.py

4. Enter your text when prompted, e.g.
   Enter your text: I am worried about my upcoming exams.
   Prediction: Stress

Example Output:
---------------
Enter your text: I feel overwhelmed with assignments.
Prediction: Stress

Enter your text: I had a great day with my friends.
Prediction: No Stress

Visualization:
--------------
The script generates a word cloud of the most common words in the dataset to help visualize the data.

Contributing:
-------------
Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

Contact:
--------
For any queries or suggestions, feel free to contact [Your Name] (your.email@example.com).

License:
--------
This project is for educational purposes only.
