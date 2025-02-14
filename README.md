### **Restaurant Recommendation System **  

## **How It Works**  
1. **Load and Preprocess Data:**  
   - The script reads restaurant data from `TripAdvisor_RestauarantRecommendation.csv`.  
   - It selects only the relevant columns: **"Name"** and **"Type"** (cuisine category).  
   - Missing values are removed to ensure data consistency.  

2. **Feature Extraction Using TF-IDF:**  
   - The **"Type"** column is transformed into numerical data using **TF-IDF (Term Frequency-Inverse Document Frequency)** to handle text-based data.  
   - **Stop words** (common words like "and," "the") are removed to improve accuracy.  

3. **Compute Similarity Using Cosine Similarity:**  
   - **Cosine Similarity** is used to measure how similar restaurants are based on their cuisine type.  
   - A similarity matrix is created to compare restaurants.  

4. **Build Recommendation Function:**  
   - The function `restaurant_recommendation(name)` retrieves similar restaurants based on the given restaurant name.  
   - It finds the **top 10** most similar restaurants and returns their names.  

5. **Example Usage:**  
   - Calling `restaurant_recommendation("Market Grill")` returns a list of similar restaurants based on cuisine type.  

---

## **Code Breakdown**  

### **1. Import Libraries**  
```python
import numpy as np
import pandas as pd
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity
```
- `numpy` and `pandas` handle data operations.  
- `text.TfidfVectorizer` converts text into numerical values.  
- `cosine_similarity` measures restaurant similarity.  

---

### **2. Load and Preprocess Data**  
```python
data = pd.read_csv("TripAdvisor_RestauarantRecommendation.csv")
print(data.head())

data = data[["Name", "Type"]]
print(data.head())

print(data.isnull().sum())  # Check for missing values
data = data.dropna()  # Remove missing values
```
- Reads data from CSV.  
- Selects restaurant **name** and **type**.  
- Removes **null values** for clean processing.  

---

### **3. Convert Text Data Using TF-IDF**  
```python
feature = data["Type"].tolist()
tfidf = text.TfidfVectorizer(input="content", stop_words="english")
tfidf_matrix = tfidf.fit_transform(feature)
similarity = cosine_similarity(tfidf_matrix)
```
- Converts restaurant **type** (e.g., "Italian, Fast Food") into numerical vectors.  
- Uses **TF-IDF** to process text while removing common stop words.  
- Computes **cosine similarity** to compare restaurants.  

---

### **4. Create Index Mapping for Restaurants**  
```python
indices = pd.Series(data.index, index=data['Name']).drop_duplicates()
```
- Maps restaurant **names to their indices** for quick lookup.  

---

### **5. Define Recommendation Function**  
```python
def restaurant_recommendation(name, similarity = similarity):
    index = indices[name]
    similarity_scores = list(enumerate(similarity[index]))
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    similarity_scores = similarity_scores[0:10]
    restaurantindices = [i[0] for i in similarity_scores]
    return data['Name'].iloc[restaurantindices]
```
- Finds **top 10 most similar restaurants** based on cosine similarity.  
- Sorts them in **descending** order (most similar first).  

---

### **6. Test the Recommendation System**  
```python
print(restaurant_recommendation("Market Grill"))
```
- Recommends restaurants similar to `"Market Grill"`.  
