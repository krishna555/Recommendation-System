import json
import random
from collections import defaultdict
import math
from itertools import combinations

class T3UserTrain:
    def make_function(self, a, b, m):
        def hash_func(x_input):
            return ((a * x_input + b) % 1000000009) % m
        return hash_func

    def main(self, sc, train_file_path, output_model_file):
        num_of_bands = 70
        numOfHashFns = 70
        
        extracted_uid_bid_rdd = sc.textFile(train_file_path)\
            .map(lambda row: json.loads(row))\
            .map(lambda row: (row["user_id"], (row["business_id"],  row["stars"])))
        
        user_id_indexed_rdd = extracted_uid_bid_rdd \
            .map(lambda record: record[0]) \
            .distinct() \
            .zipWithIndex()
        
        user_id_to_index_map = user_id_indexed_rdd.collectAsMap()
        index_to_uid_map = {v: k for k, v in user_id_to_index_map.items()}

        business_id_indexed_rdd = extracted_uid_bid_rdd \
            .map(lambda record: record[1][0]) \
            .distinct() \
            .zipWithIndex()
        
        business_id_to_index_map = business_id_indexed_rdd.collectAsMap()
        index_to_bid_map = {v: k for k, v in business_id_to_index_map.items()}

        hash_fns = []
        buckets = len(user_id_to_index_map)
        for i in range(numOfHashFns):
            a = random.randrange(1, 972666949, pow(10, 7))
            b = random.randrange(1, 972667571, pow(10, 7))
            hash_fns.append(self.make_function(a, b, buckets))
        
        uid_tup_bid_rating = extracted_uid_bid_rdd\
            .map(lambda record: (user_id_to_index_map[record[0]], (business_id_to_index_map[record[1][0]], record[1][1])))\
            .groupByKey()\
            .map(lambda record: (record[0], list(record[1])))\
            .filter(lambda tuple: len(tuple[1]) >= 3)
        
        uid_tup_bid_rating_collected = uid_tup_bid_rating.collect()
        
        uid_bid_rtg_map = defaultdict(dict)
        for tup_uid_tup_bid_rating in uid_tup_bid_rating_collected:
            uid = tup_uid_tup_bid_rating[0]
            bid_ratings = tup_uid_tup_bid_rating[1]
            for bid_rating in bid_ratings:
                bid = bid_rating[0]
                rating = bid_rating[1]
                uid_bid_rtg_map[uid][bid] = rating
        
        # tup_bid_uid_rdd = extracted_uid_bid_rdd.map(lambda tup: (business_id_to_index_map[tup[1][0]], user_id_to_index_map[tup[0]]))

        tup_bid_uid_rdd = uid_tup_bid_rating.flatMap(lambda tup: [(bid_rating[0], tup[0]) for bid_rating in tup[1]])
        bid_to_hash = defaultdict(list)
        for item in business_id_to_index_map.items():
            bid = item[0]
            index = item[1]
            n_hashes = [func(index) for func in hash_fns]
            bid_to_hash[index] = n_hashes

        def chooseLeastHashValue(list1, list2):
            return [min(val1, val2) for (val1, val2) in zip(list1, list2)]

        # Needs Validation
        sig_matrix_uid_n_hash_list = tup_bid_uid_rdd\
                .map(lambda tup_bid_uid: (tup_bid_uid[0], tup_bid_uid[1]))\
                .groupByKey().mapValues(lambda user_id_list: list(set(user_id_list)))\
                .flatMap(lambda bid_uid_list: [(user_id, bid_to_hash[bid_uid_list[0]]) for user_id in bid_uid_list[1]])\
                .reduceByKey(chooseLeastHashValue).collect()

        candidates = set()

        for band_num in range(num_of_bands):
            bkt = defaultdict(set)
            for tup_uid_list_hash in sig_matrix_uid_n_hash_list:
                uid = tup_uid_list_hash[0]
                hash_list = tup_uid_list_hash[1]
                bkt[hash(hash_list[band_num])].add(uid)
            for uids in bkt.values():
                if len(uids) > 1:
                    for pair in combinations(uids, 2):
                        candidates.add(tuple(sorted(pair)))

        def get_pearson_sim(bid_rtg_map1, bid_rtg_map2):
            co_rated_items_list = set(bid_rtg_map1.keys()) & set(bid_rtg_map2.keys())
            co_rated_items_1 = []
            for key in co_rated_items_list:
                co_rated_items_1.append(bid_rtg_map1[key])
            co_rated_items_2 = []
            for key in co_rated_items_list:
                co_rated_items_2.append(bid_rtg_map2[key])
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
        
        def jaccard_sim(b1, b2):
            a = set(b1)
            b = set(b2)
            return len(a & b) / len(a | b) 

        with open(output_model_file, 'w+') as of:
            for candidate_pair in candidates:
                user1 = candidate_pair[0]
                user2 = candidate_pair[1]
                b1 = uid_bid_rtg_map[user1]
                b2 = uid_bid_rtg_map[user2]
                b1_keys = b1.keys()
                b2_keys = b2.keys()
                if len(set(b1_keys) & set(b2_keys)) >= 3:
                    js = jaccard_sim(b1_keys, b2_keys)
                    if js >= 0.01:
                        ps = get_pearson_sim(b1, b2)
                        if ps > 0:
                            of.write(json.dumps({'u1': index_to_uid_map[user1], 'u2': index_to_uid_map[user2], 'sim': ps}) + '\n')
        of.close()
