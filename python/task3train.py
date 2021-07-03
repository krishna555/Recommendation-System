import sys
from pyspark import SparkConf, SparkContext
from t3_user_train import T3UserTrain
from t3_item_train import T3ItemTrain
class task3:
    def main(self):
        conf = SparkConf().setAppName("task3").setMaster("local[*]")
        sc = SparkContext(conf=conf).getOrCreate()
        sc.setLogLevel("OFF")
        train_input_path = sys.argv[1]
        output_file_path = sys.argv[2]
        cf_type = sys.argv[3]
        if cf_type == "item_based":
            # Do Something
            t3_item_train = T3ItemTrain()
            t3_item_train.main(sc, train_input_path, output_file_path)
        else:
            # Do User Based Stuff
            t3_user_train = T3UserTrain()
            t3_user_train.main(sc, train_input_path, output_file_path)

if __name__ == "__main__":
    t3 = task3()
    t3.main()