import pandas as pd
import spacy
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
import numpy as np
from typing import List, Dict, Tuple
import json

class ResumeProcessor:
    def __init__(self):
        # Load spaCy model with custom pipeline
        self.nlp = spacy.load("en_core_web_lg")
        
        # Initialize skill synonyms dictionary
        self.skill_synonyms = {
            "python": ["python3", "python2", "py"],
            "javascript": ["js", "ecmascript", "node.js", "nodejs"],
            "machine learning": ["ml", "deep learning", "ai", "artificial intelligence"],
            "sql": ["mysql", "postgresql", "oracle", "database"],
        }
        
        # Common technical skills and their weights
        self.technical_skills = {
            "python": 1.5,
            "javascript": 1.3,
            "java": 1.3,
            "c++": 1.3,
            "machine learning": 1.4,
            "sql": 1.2,
            "aws": 1.3,
            "docker": 1.2,
        }

    def extract_experience(self, text: str) -> int:
        """Extract years of experience from text."""
        experience_pattern = r'(\d+)[\s-]*(?:year|yr)s?\s+(?:of\s+)?experience'
        matches = re.findall(experience_pattern, text.lower())
        return sum(int(match) for match in matches) if matches else 0

    def extract_skills(self, text: str) -> List[str]:
        """Extract technical skills from text."""
        doc = self.nlp(text.lower())
        skills = []
        
        # Extract skills using pattern matching and synonym dictionary
        for skill, synonyms in self.skill_synonyms.items():
            if skill in text.lower() or any(syn in text.lower() for syn in synonyms):
                skills.append(skill)
        
        # Extract additional technical terms using spaCy's entity recognition
        for ent in doc.ents:
            if ent.label_ in ["PRODUCT", "ORG"] and ent.text.lower() in self.technical_skills:
                skills.append(ent.text.lower())
        
        return list(set(skills))

    def calculate_skill_score(self, resume_skills: List[str], job_skills: List[str]) -> float:
        """Calculate weighted skill match score."""
        if not job_skills:
            return 0.0
        
        matched_skills = set(resume_skills) & set(job_skills)
        total_weight = sum(self.technical_skills.get(skill, 1.0) for skill in matched_skills)
        max_possible_weight = sum(self.technical_skills.get(skill, 1.0) for skill in job_skills)
        
        return total_weight / max_possible_weight if max_possible_weight > 0 else 0.0

class ResumeFilter:
    def __init__(self):
        self.processor = ResumeProcessor()
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2
        )

    def preprocess_text(self, text: str) -> str:
        """Advanced text preprocessing."""
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Process with spaCy
        doc = self.processor.nlp(text)
        
        # Keep only relevant tokens
        tokens = [
            token.lemma_ for token in doc 
            if not token.is_stop and not token.is_punct and len(token.text) > 2
        ]
        
        return ' '.join(tokens)

    def calculate_match_score(self, resume: str, job_description: str) -> Dict:
        """Calculate comprehensive match score between resume and job description."""
        # Preprocess texts
        processed_resume = self.preprocess_text(resume)
        processed_job = self.preprocess_text(job_description)
        
        # Extract features
        resume_experience = self.processor.extract_experience(resume)
        job_experience = self.processor.extract_experience(job_description)
        
        resume_skills = self.processor.extract_skills(resume)
        job_skills = self.processor.extract_skills(job_description)
        
        # Calculate TF-IDF similarity
        tfidf_matrix = self.vectorizer.fit_transform([processed_resume, processed_job])
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        
        # Calculate skill match score
        skill_score = self.processor.calculate_skill_score(resume_skills, job_skills)
        
        # Calculate experience match score
        exp_score = min(1.0, resume_experience / max(1, job_experience)) if job_experience > 0 else 1.0
        
        # Calculate final weighted score
        final_score = (
            0.4 * cosine_sim +  # Overall content similarity
            0.4 * skill_score +  # Technical skill match
            0.2 * exp_score      # Experience match
        )
        
        return {
            "total_score": round(final_score * 100, 2),
            "content_similarity": round(cosine_sim * 100, 2),
            "skill_match": round(skill_score * 100, 2),
            "experience_match": round(exp_score * 100, 2),
            "matched_skills": list(set(resume_skills) & set(job_skills)),
            "missing_skills": list(set(job_skills) - set(resume_skills)),
            "years_of_experience": resume_experience
        }

def main():
    # Initialize the filter
    resume_filter = ResumeFilter()
    
    # Sample data (replace with your actual data)
    resumes = [
        """Senior Software Engineer with 7 years of experience in Python, JavaScript, and AWS.
        Led development of machine learning applications using TensorFlow and scikit-learn.
        Implemented CI/CD pipelines using Docker and Jenkins.""",
        
        """Data Scientist with 3 years of experience in Python, R, and SQL.
        Developed predictive models using machine learning algorithms.
        Experience with data visualization using Tableau."""
    ]
    
    job_description = """
    Senior Software Engineer position requiring 5+ years of experience.
    Strong knowledge of Python, JavaScript, and AWS required.
    Experience with machine learning and Docker is a plus.
    Must have experience with CI/CD practices.
    """
    
    # Process each resume
    for i, resume in enumerate(resumes, 1):
        results = resume_filter.calculate_match_score(resume, job_description)
        
        print(f"\nResults for Resume {i}:")
        print(f"Overall Match Score: {results['total_score']}%")
        print(f"Content Similarity: {results['content_similarity']}%")
        print(f"Skill Match: {results['skill_match']}%")
        print(f"Experience Match: {results['experience_match']}%")
        print(f"Matched Skills: {', '.join(results['matched_skills'])}")
        print(f"Missing Skills: {', '.join(results['missing_skills'])}")
        print(f"Years of Experience: {results['years_of_experience']}")

if __name__ == "__main__":
    main()
