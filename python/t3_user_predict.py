import sys
import json
from collections import defaultdict
class T3UserPredict:
    def main(self, sc, train_file, test_file, model_file, output_file):
        train_rdd = sc.textFile(train_file).map(lambda row: json.loads(row))
        test_rdd = sc.textFile(test_file).map(lambda row: json.loads(row))
        model_rdd = sc.textFile(model_file).map(lambda row: json.loads(row))
        
        extracted_train_data = train_rdd.map(lambda record: (record["business_id"], (record["user_id"], record["stars"])))

        # uid_to_sum_cnt_map = extracted_train_data\
        #         .map(lambda record: (record[1][0], record[1][1]))\
        #         .groupByKey()\
        #         .mapValues(lambda stars: (sum([float(num) for num in stars]), len(stars))).collectAsMap()
        
        # Key: Business_id, Value: List: UserID, Rating
        bid_aggr_train_data = extracted_train_data\
            .groupByKey()\
            .mapValues(lambda user_stars_tuples: list(set(user_stars_tuples)))
        
        # Key: User_ID, Value: List: Tuple(Business_id, stars)
        uid_tup_bid_rating = extracted_train_data\
            .map(lambda record: (record[1][0], (record[0], record[1][1])))\
            .groupByKey()\
            .map(lambda record: (record[0], list(record[1]))).collect()

        # Key: UserId Value: (Sum, Count)
        uid_to_sum_cnt_map = {}
        # Map[UID][BID] = Rating
        uid_bid_rtg_map = defaultdict(dict)
        for tup_uid_tup_bid_rating in uid_tup_bid_rating:
            uid = tup_uid_tup_bid_rating[0]
            bid_ratings = tup_uid_tup_bid_rating[1]
            num_ratings = len(bid_ratings)
            sum_ratings = 0
            for bid_rating in bid_ratings:
                bid = bid_rating[0]
                rating = bid_rating[1]
                sum_ratings += float(rating)
                uid_bid_rtg_map[uid][bid] = rating
            uid_to_sum_cnt_map[uid] = (sum_ratings, num_ratings)

        # Key: Business_id, Value: UserID
        extracted_test_data = test_rdd.map(lambda record: (record["business_id"], record["user_id"]))

        extracted_model_data = model_rdd.map(lambda record: ((record['u1'], record['u2']), record['sim'])).collect()

        model = {}
        for tup in extracted_model_data:
            user_pair = tup[0]
            sim = tup[1]
            model[tuple(sorted(user_pair))] = sim

        def predictor(record, model, uid_bid_rtg_map, uid_to_sum_cnt_map):
            # Key: Tuple (business_id, user_id), Value: List(tuple(user_id, stars))
            bid = record[0][0]
            uid1 = record[0][1]
            neighbours = []
            N = 10
            for uid_stars_tup in record[1]:
                uid2 = uid_stars_tup[0]
                rating = uid_stars_tup[1]
                item_pair = tuple(sorted([uid1, uid2]))
                if model.get(item_pair):
                    neighbours.append((rating, model[item_pair], uid2))
            
            neighbours.sort(key=lambda x: -x[1])
            neighbours = neighbours[:N]
            num_sum = 0
            den_sum = 0

            for rating_sim_uid_tup in neighbours:
                rtg = rating_sim_uid_tup[0]
                sim = rating_sim_uid_tup[1]
                uid = rating_sim_uid_tup[2]
                sum_and_cnt_tup = uid_to_sum_cnt_map[uid]
                uid_sum = sum_and_cnt_tup[0]
                uid_cnt = sum_and_cnt_tup[1]
                avg_term = 0
                if uid_cnt - 1 > 0:
                    avg_term = (uid_sum - rtg) / (uid_cnt - 1)
                num_sum += (rtg - avg_term) * sim
                den_sum += abs(sim)
            
            term2 = 0
            if num_sum != 0 and den_sum != 0:
                term2 = num_sum / den_sum
            
            # current_term_rating = uid_bid_rtg_map[uid1][bid] if bid in uid_bid_rtg_map[uid1] else 0
            uid1_sum, uid1_cnt = uid_to_sum_cnt_map[uid1]
            avg_term_curr = uid1_sum/uid1_cnt
            # if uid1_cnt - 1 > 0:
            #     avg_term_curr = (uid1_sum - current_term_rating) / (uid1_cnt - 1)

            return (uid1, bid, avg_term_curr + term2)


        # Train Key: Business_id, Value: List: UserID, Rating
        # Test Key: Business_id, Value: UserID
        predictions = bid_aggr_train_data.join(extracted_test_data)\
            .map(lambda tup: ((tup[0], tup[1][1]), tup[1][0]))\
            .filter(lambda tup: tup[1][0] != tup[0][1])\
            .groupByKey()\
            .flatMapValues(lambda val: list(val))\
            .map(lambda record: predictor(record, model, uid_bid_rtg_map, uid_to_sum_cnt_map))\
            .filter(lambda tup: tup[2] != 0)\
            .collect()
        
        with open(output_file, 'w+') as of:
            for prediction in predictions:
                of.write(json.dumps({'user_id': prediction[0], 'business_id': prediction[1], 'stars': prediction[2]}) + '\n')
        of.close()
