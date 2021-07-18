# Recommendation-System

Two primary tasks have been implemented in this project:

1. Build a content-based recommendation system by generating profiles from review texts for users and businesses. TF-IDF was used to measure word importance in review texts to create business profile vector. During prediction, we estimate if a user would prefer to review a business by computing cosine distance between profile vectors.
   
   1. Outcome: Precision was found to be 1.0 and recall was found to be 0.96.

2. Build a collaborative filtering Recommendation systems - Item-based and User-based Collaborative filtering recommendation systems.
   
   1. Outcome: RMSE of Item-based collaborative filtering recommendation system was found to be 0.85 and RMSE of USer-based Collaborative Filtering Recommendation system was found to be 0.96.
