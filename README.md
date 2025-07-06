# music_recommendation_system
A content-based music recommendation system built using machine learning techniques. This project uses a Spotify dataset from Kaggle to recommend songs based on audio features and user preferences. Implemented with Python and popular ML libraries to provide personalized song suggestions.

# 🎧 Music Recommendation System

![Python](https://img.shields.io/badge/Python-3.8+-blue?logo=python)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Used-orange?logo=scikit-learn)
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)
![Model-Type](https://img.shields.io/badge/Model-Content--Based-blue)
![Status](https://img.shields.io/badge/Project-Complete-brightgreen)

A content-based music recommendation system using a Spotify dataset from Kaggle. It recommends songs based on their similarity in **artist**, **genre**, **album**, and **user rating**.

---

## 📂 Dataset

- The dataset (`ex.csv`) includes the following columns:
  - `Song-Name`
  - `Singer/Artists`
  - `Genre`
  - `Album/Movie`
  - `User-Rating`

---

## 🧠 Approach

1. **Data Cleaning**:
   - Removed nulls and duplicates
   - Normalized text: stripped spaces, commas

2. **Feature Engineering**:
   - Combined `Singer/Artists`, `Genre`, `Album/Movie`, and `User-Rating` into a new column: `tags`

3. **Vectorization**:
   - Used **CountVectorizer** to convert `tags` into a numerical format

4. **Similarity Calculation**:
   - Used **cosine similarity** to find and recommend the top-N most similar songs

---

## 🛠️ Technologies Used

- Python  
- Pandas  
- Scikit-learn  
- NumPy  

---

## 🚀 How to Run

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/music-recommendation-system.git
   cd music-recommendation-system
