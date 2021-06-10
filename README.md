# tweet-classification
 
### What is TFIDF?
- Since machine learning algorithms deal with numbers, text data needs to be transformed into numbers. This process is known as text vectorization.  
- The simplest way to perform text vectorization is to simply create a vector for each data sample that counts the frequency of each word in the entire dataset.  
- This method is called Count Vectorization, the problem with it is that it can't identify words that are more or less important for analysis. It just considers more frequent words as the most statistically significant word.

Term Frequency - Inverse Document Frequency or TF-IDF solves this problem by providing a numerical representation of the importance of a given word for statistical analysis. 
- TF-IDF measures how relevant a word is by using two metrics:  
  - Term Frequency - How many times does a word appear in a given sample?
  - Inverse Document Frequency - Where document frequency measures how many times does a word appear in the entire set of samples.
- As a result, words that are too common or too rare in the corpus are penalized by giving them lower TF-IDF scores.
