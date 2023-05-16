import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
        
        
class Recommendations:

    def __init__(self, K = None):
        self.num_click = None
        self.num_features = None
        self.K = K
        self.history = []
     
    ## Calculate item-item cosine similarity ##   
    def calc_item_similarities(self, train):
        #Remove "Rating"
        train_sim = train.drop(["Rating"], axis=1)
        M = train_sim.to_numpy()
        # Calculate cosine similarity
        Item_similarities = cosine_similarity(M, M)
        return Item_similarities

    ## Return top K popular items based on “Rating” ##
    def topk_popular(self, train):
        topK_index = train.nlargest(self.K,"Rating").index
        topK_popular_car = train.iloc[topK_index]
        return topK_index, topK_popular_car
    
    ## Return the Recommended items based on item-item similarity ##
    # To Prevent Overspecialization
    # 1. (1-popularization)*K of K recommended item are based on item similarity.
    # 2. (popularization*K) of K recommended item are based on popular items 
    # 3. popularization = 0.2 by default
    def calc_rating_item(self, row_clicked, Item_similarities, train, popularization=0.2):

        self.click_history(row_clicked)

        K_similar = np.int_(np.round(self.K*(1-popularization)))    

        sorted_item = np.argsort(Item_similarities[row_clicked])[::-1]
        # Prevent to recommend the same item that has been clicked by the customer
        for i in self.history:
            sorted_item = np.delete(sorted_item, np.where(sorted_item == i), axis=0)
        most_similar_item = sorted_item[:K_similar]

        # Look for popular item based on the rest of the data set, excluding most_similar_item 
        index_rest = sorted_item[K_similar:]
        train_rest = train.iloc[index_rest]

        # For more than a single click
        if len(self.history) > 1:
            # Based on clicking history,
            # get a dictionary that contains the top 5 frequent features as the key with the most frequent values in each feature as the value.
            feature_key_top = self.valued_feature(train)
            query = ' and '.join([f'{k} == {repr(v)}' for k, v in feature_key_top.items()])
            # Update data set based on feature_key_top 
            train_rest = train_rest.query(query)
        popular_item = train_rest.nlargest(self.K-K_similar,"Rating").index
        # Combine the indexes of most_similar_item & popular_item
        recommended_item = np.concatenate((most_similar_item, popular_item), axis=0)    
        return recommended_item


    ## Store the clicking history of the customer ##
    def click_history(self, row_clicked):
        self.history.append(row_clicked)
        print("click_history", self.history)
    
    ## Based on the click_history()
    ## get a dictionary that contains the top 5 frequent features as the key with the most frequent values in each feature as the value.
    def valued_feature(self, train):
        # Get clicking history data
        train_history = train.iloc[self.history]
        train_history = train_history.drop(["Rating"], axis=1)

        feature_key = {}
        feature_value = {}
        for col in train_history.columns:
            count_dict = train_history[col].value_counts().to_dict()
            max_value = max(count_dict.values())
            # The most frequent value under each feature
            feature_value[col] = max_value
            # The features with the most frequent count of a certain value 
            feature_key[col] = [k for k, v in count_dict.items() if v == max_value][0]
        
        feature_value_top5 = sorted(feature_value, key=feature_value.get, reverse=True)[:5]
        # feature_top = [f for f in feature_value_top5 if feature_value[f] > len(self.history)/2]
        feature_key_top = {k:feature_key[k] for k in feature_value_top5 if k in feature_key}
        print("feature_key_top = {}".format(feature_key_top))
        return feature_key_top
           







