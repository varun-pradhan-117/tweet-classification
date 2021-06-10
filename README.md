# tweet-classification
 
### **What is TFIDF?**
- Since machine learning algorithms deal with numbers, text data needs to be transformed into numbers. This process is known as text vectorization.  
- The simplest way to perform text vectorization is to simply create a vector for each data sample that counts the frequency of each word in the entire dataset.  
- This method is called Count Vectorization, the problem with it is that it can't identify words that are more or less important for analysis. It just considers more frequent words as the most statistically significant word.

Term Frequency - Inverse Document Frequency or TF-IDF solves this problem by providing a numerical representation of the importance of a given word for statistical analysis. 
- TF-IDF measures how relevant a word is by using two metrics:  
  - Term Frequency - How many times does a word appear in a given sample?
  - Inverse Document Frequency - Where document frequency measures how many times does a word appear in the entire set of samples.
- As a result, words that are too common or too rare in the corpus are penalized by giving them lower TF-IDF scores.
- TFIDF is calculated as:  
  - tfidf(t,d,D) = tf(t,d) * idf(t,D)  
  - Where tf(t,d) represents the frequency of a term in a document and  
    idf(t,D) represents the frequency of a term in the entire set of documents  
- tf(t,d) and idf(t,D) can be weighted in various different ways. The sklearn implementation just uses the number of times a term occurs in a document for tf, without weighting. While idf(t,D) is calculated as:  
  - ![idf calculation](images/idf-calc.JPG)
- This differs from the standard definition of idf:  
  - ![idf OG](images/idf-OG.jpg)
- The 1 is added to the denominator to ensure that log(0) isn't calculated for a completely new term. 

To compare TF-IDF and Count Vectorizer, we can take a basic example.  
- Say a document has the following 2 sentences:  
![sample docs](images/vectorizer-example.jpg)
- If we perform count vectorization we get the following matrix:  
![count vect sample](images/count-vect-example.jpg)
- If we perform tfidf vectorization we get the following matrix:  
![tfidf sample](images/tfidf-example.jpg)
- As we can see, tfidf vectorization gives us more information as compared to the Count Vectorization.  
- It is important to note that tf-idf values for small text corpuses are likely to be noisy.


### **Difference between Stemming and Lemmatization** 
