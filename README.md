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
- Stemming and Lemmatization are used to find the "stem" or "root" of a given word.
- In a given corpus, for grammatical reasons, a word might be used in its many different forms. For example: eat, eaten, eating, eats, ate etc.
- Stemming and Lemmatization is performed so that these words aren't treated as completely different words.
- Stemming algorithms follow a specific set of rules and cut off suffixes from the beginning or end of the word to reach the "stem" words. For example Studies and studying are both stemmed to studi.
- Lemmatization is a more complicated process and it involves an actual morphological analysis of the word. It returns the base or dictionary form of a word called a lemma.
- For example in case of the word "saw" a stemmer might return just "s" or "saw" as is, but a lemmatizer will return "saw" or "see based on whether it was used in the context as a noun or a verb.
- As a result, lemmatization tends to be more accurate and better conveys what the word could mean but it is also more computationally expensive.
- Therefore in tasks where accuracy is more important, lemmatization usually performs better.
- But in some cases such as indexing a massive number of documents in a search engine, stemming might be preferred as it is much faster, and much easier to implement.
- The difference between the two can be seen in the following image:
![l-s-compare](images/lemma-vs-stem.JPG)
- As we can see, lemmatization tries to maintain more information while stemming fails to notice that "United States" is a proper noun.

### Basic model with just TF-IDF vectorization  
Two models were made with just TF-IDF vectorization
- First model set a limit on the vector length=length of longest sentence in the training data.
  - It had an accuracy of 68% on the test set
  - ![limit-tfidf-accuracy](images/limit-tfidf-accuracy.jpg)
  - ![limit-tfidf-loss](images/limit-tfidf-loss.jpg)
  - As we can see accuracy on the test set and training set remains around the same after the 5th epoch without any major spikes.
- In the 2nd model,no limit was set on the vector length and the model was allowed to create vectors for all words in the vocabulary
  - It had an accuracy of 83.8% on the test set 
  - ![no-limit-tfidf-accuracy](images/no-limit-tfidf-accuracy.jpg)
  - ![no-limit-tfidf-loss](images/n0-limit-tfidf-loss.jpg)
  - As we can see, training accuracy keeps increasing while accuracy on test set remains the same. This overfitting can be fixed through early stopping.
  
### Model with TF-IDF weighted embedding
- Another model is made where TF-IDF vectors are generated for the data.
- The GloVe vectors for all TF-IDF terms are obtained using spacy
- The TF-IDF vectors are multiplied by the GloVe vectors for those TF-IDF terms to obtain TF-IDF weighted embeddings for the data
- It had an accuracy of 72.1% on the test set
- ![weighted accuracy](images/weighted-accuracy.jpg)
- ![weighted loss](images/weighted-loss.jpg)


### **LSTM**
