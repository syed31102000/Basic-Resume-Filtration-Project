Resume Matching System

This project is a comprehensive resume matching system designed to evaluate the compatibility between resumes and job descriptions. The system uses natural language processing (NLP) techniques,
such as named entity recognition (NER), tokenization, and lemmatization, to extract and compare key components such as experience, skills, and overall content. The aim is to generate a match score 
based on these extracted features and assess how well a resume aligns with the job requirements.

Tools and Libraries:
SpaCy:
It is utilized for NLP tasks such as named entity recognition (NER), tokenization, and lemmatization. It helps in extracting relevant technical skills and other significant terms from the text.
scikit-learn:

TfidfVectorizer:
It is used for converting the text into numerical vectors based on Term Frequency-Inverse Document Frequency (TF-IDF). It helps in comparing the semantic similarity between a resume and a job description.

NLTK:
Used for text preprocessing and tokenization, as well as to gather synonyms for technical skills in resumes.

Regular Expressions (re):
Used for pattern matching to extract experience from the resume and job description texts.

Python (Data Science Libraries):
NumPy: For handling data and calculations.
JSON: For structuring and outputting results.
