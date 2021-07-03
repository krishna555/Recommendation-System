import json
import random
from collections import defaultdict
import math
from itertools import combinations

class T3ItemTrain:
    def main(self, sc, train_file_path, output_model_file):
        extracted_train_rdd = sc.textFile(train_file_path).map(lambda row: json.loads(row)).map(lambda row: (row["user_id"], (row["business_id"], row["stars"])))

        user_id_indexed_rdd = extracted_train_rdd \
            .map(lambda record: record[0]) \
            .distinct() \
            .zipWithIndex()
        
        user_id_to_index_map = user_id_indexed_rdd.collectAsMap()

        business_id_indexed_rdd = extracted_train_rdd \
            .map(lambda record: record[1][0]) \
            .distinct() \
            .zipWithIndex()
        
        business_id_to_index_map = business_id_indexed_rdd.collectAsMap()
        index_to_bid_map = {v: k for k, v in business_id_to_index_map.items()}

        bid_tup_uid_rating = extracted_train_rdd\
            .map(lambda record: (business_id_to_index_map[record[1][0]], (user_id_to_index_map[record[0]], record[1][1])))\
            .groupByKey()\
            .map(lambda record: (record[0], list(record[1])))\
            .filter(lambda tuple: len(tuple[1]) >= 3).collect()
        
        bid_uid_rtg_map = defaultdict(dict)
        for tup_bid_tup_uid_rating in bid_tup_uid_rating:
            bid = tup_bid_tup_uid_rating[0]
            uid_ratings = tup_bid_tup_uid_rating[1]
            for uid_rating in uid_ratings:
                uid = uid_rating[0]
                rating = uid_rating[1]
                bid_uid_rtg_map[bid][uid] = rating

        def get_pearson_sim(uid_rtg_map1, uid_rtg_map2):
            co_rated_items_list = set(uid_rtg_map1.keys()) & set(uid_rtg_map2.keys())
            co_rated_items_1 = []
            for key in co_rated_items_list:
                co_rated_items_1.append(uid_rtg_map1[key])
            co_rated_items_2 = []
            for key in co_rated_items_list:
                co_rated_items_2.append(uid_rtg_map2[key])
            avg1 = sum(co_rated_items_1) / len(co_rated_items_1)
            avg2 = sum(co_rated_items_2) / len(co_rated_items_2)
            normalized_rtgs_1 = []
            for rating in co_rated_items_1:
                normalized_rtgs_1.append(rating - avg1)
            normalized_rtgs_2 = []
            for rating in co_rated_items_2:
                normalized_rtgs_2.append(rating - avg2)
            num_sum = 0
            den_sum1 = 0
            den_sum2 = 0
            for rtg1, rtg2 in zip(normalized_rtgs_1, normalized_rtgs_2):
                num_sum += (rtg1 * rtg2)
                den_sum1 += (rtg1 * rtg1)
                den_sum2 += (rtg2 * rtg2)
            den = math.sqrt(den_sum1) *math.sqrt(den_sum2)
            
            if num_sum > 0 and den > 0:
                return num_sum / den
            else:
                return 0

        with open(output_model_file, "w+") as of:
            bids = business_id_to_index_map.values()
            for business_id_pair in combinations(bids, 2):
                bid1 = business_id_pair[0]
                bid2 = business_id_pair[1]
                uid1 = bid_uid_rtg_map[bid1]
                uid2 = bid_uid_rtg_map[bid2]
                uid1_keys = uid1.keys()
                uid2_keys = uid2.keys()
                if len(set(uid1_keys) & set(uid2_keys)) >= 3:
                    ps = get_pearson_sim(uid1, uid2)
                    if ps > 0:
                        of.write(json.dumps({'b1': index_to_bid_map[bid1], 'b2': index_to_bid_map[bid2], 'sim': ps}) + '\n')
        of.close()