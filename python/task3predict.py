import sys
from pyspark import SparkConf, SparkContext
from t3_user_predict import T3UserPredict
from t3_item_predict import T3ItemPredict
class Task3Predict:
    def main(self):
        conf = SparkConf().setAppName("task3").setMaster("local[*]")
        sc = SparkContext(conf=conf).getOrCreate()
        sc.setLogLevel("OFF")
        train_file = sys.argv[1]
        test_file = sys.argv[2]
        model_file = sys.argv[3]
        output_file = sys.argv[4]
        cf_type = sys.argv[5]
        if cf_type == "item_based":
            # Do Something
            t3_item_predict = T3ItemPredict()
            t3_item_predict.main(sc, train_file, test_file, model_file, output_file)
        else:
            # Do User Based Stuff
            t3_user_predict = T3UserPredict()
            t3_user_predict.main(sc, train_file, test_file, model_file, output_file)

if __name__ == "__main__":
    t3 = Task3Predict()
    t3.main()