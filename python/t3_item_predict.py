import sys
import json
from collections import defaultdict
class T3ItemPredict:
    def main(self, sc, train_file, test_file, model_file, output_file):
        train_rdd = sc.textFile(train_file).map(lambda row: json.loads(row))
        test_rdd = sc.textFile(test_file).map(lambda row: json.loads(row))
        model_rdd = sc.textFile(model_file).map(lambda row: json.loads(row))

        # extracted_train_data = train_rdd.map(lambda record: (record["user_id"], (record["business_id"], record["stars"])))\
        #     .groupByKey()\
        #     .mapValues(lambda user_stars_tuples: list(set(user_stars_tuples)))
         
        extracted_train_data = train_rdd.map(lambda record: ((record["user_id"], record["business_id"]), record["stars"]))\
            .groupByKey()\
            .mapValues(lambda stars: sum([float(num) for num in stars]) / len(stars))\
            .map(lambda record: (record[0][0], (record[0][1], record[1])))\
            .groupByKey()\
            .mapValues(lambda value: list(set(value)))
        
        extracted_test_data = test_rdd.map(lambda record: (record["user_id"], record["business_id"]))

        extracted_model_data = model_rdd.map(lambda record: ((record['b1'], record['b2']), record['sim'])).collect()
        model = {}
        for tup in extracted_model_data:
            business_pair = tup[0]
            sim = tup[1]
            model[tuple(sorted(business_pair))] = sim

        def predictor(record, model):
            # Key: Tuple (user_id, businesss_id), Value: List(tuple(business_id, stars))
            uid = record[0][0]
            bid = record[0][1]
            neighbours = []
            N = 10
            for bid_stars_tup in record[1]:
                bid2 = bid_stars_tup[0]
                rating = bid_stars_tup[1]
                item_pair = tuple(sorted([bid, bid2]))
                if model.get(item_pair):
                    neighbours.append((rating, model[item_pair]))
            
            neighbours.sort(key=lambda x: -x[1])
            neighbours = neighbours[:N]
            num_sum = 0
            den_sum = 0
            for rating_sim_pair in neighbours:
                num_sum += (rating_sim_pair[0] * rating_sim_pair[1])
                den_sum += abs(rating_sim_pair[1])
            
            if num_sum != 0 and den_sum != 0:
                stars = num_sum / den_sum
            else:
                stars = 0
            if stars < 0:
                stars = 0
            return (uid, bid, stars)

        predictions = extracted_train_data.join(extracted_test_data)\
                .map(lambda tup: ((tup[0], tup[1][1]), tup[1][0]))\
                .filter(lambda tup: tup[1][0] != tup[0][1])\
                .groupByKey()\
                .flatMapValues(lambda val: list(val))\
                .map(lambda record: predictor(record, model))\
                .filter(lambda tup: tup[2] != 0) \
                .collect()
        
        with open(output_file, 'w+') as of:
            for prediction in predictions:
                of.write(json.dumps({'user_id': prediction[0], 'business_id': prediction[1], 'stars': prediction[2]}) + '\n')
        of.close()