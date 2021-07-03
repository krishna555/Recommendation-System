import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.json4s.jackson.JsonMethods._
import org.json4s._
import org.json4s.jackson.Serialization

import java.io.{File, PrintWriter}
import scala.io.Source

object task2train {
    def main(args: Array[String]): Unit = {
        val conf: SparkConf = new SparkConf().setAppName("task2").setMaster("local[*]")
        val sc: SparkContext = new SparkContext(conf)
        sc.setLogLevel("OFF")
        val trainFilePath = args(0)
        val modelFilePath = args(1)
        val stopwordsPath = args(2)
        val extracted_data_rdd = sc.textFile(trainFilePath)
            .map((row) => parse(row))
            .map((record) => {
                implicit val formats = DefaultFormats
                ((record \ "user_id").extract[String], (record \ "business_id").extract[String], (record \ "text").extract[String])
            })
        val stopWordsSet = Source.fromFile(stopwordsPath).getLines.map((line) => line.trim()).toSet

        val user_index_rdd = extracted_data_rdd.map((record) => record._1).distinct().zipWithIndex()
        val bid_index_rdd = extracted_data_rdd.map((record) => record._2).distinct().zipWithIndex()
        val user_index_dict = user_index_rdd.collectAsMap()
        val bid_index_dict = bid_index_rdd.collectAsMap()
        val index_to_bid_map = bid_index_dict.map(_.swap)
        val index_to_uid_map = user_index_dict.map(_.swap)

        val bid_wds_rdd = extracted_data_rdd.map((record) => {
            (bid_index_dict(record._2), record._3)
        }).groupByKey().mapValues((reviews_list) => {
            val texts = reviews_list.toList
            val wordLists = texts.flatMap((text) => {
                val filtered_text = text.toLowerCase().replaceAll("""[\p{Punct}]""", "")
                filtered_text.split("""[\n\s\r]+""")
            })
            wordLists.filter((word) => word != null && word != "" && !stopWordsSet.contains(word) && word.matches("[a-zA-Z]+"))
        })

        val bid_wds_tf = bid_wds_rdd.mapValues((wordList) => {
                val wc_map = scala.collection.mutable.HashMap.empty[String, Int]
                var highest = -1
                for (word <- wordList) {
                    val n = wc_map.getOrElse(word, 0)
                    wc_map += (word -> (n + 1))
                    if (n + 1 > highest) {
                        highest = n + 1
                    }
                }
                wc_map.view.map { case (word, cnt) => (word, cnt / highest.toDouble)} toList
            })
            .flatMap((record) => {
                record._2.map((kv) => (record._1, kv._1, kv._2))
            })
        val num_docs = bid_index_dict.size
        val idf_vals = bid_wds_rdd.flatMap((tup) => {
            tup._2.map((word) => {
                (word, tup._1)
            })
        }).groupByKey()
            .mapValues(tup => {
                var log2 = (x: Double) => scala.math.log10(x)/scala.math.log10(2.0)
                log2(num_docs / tup.toSet.size)
            }).collectAsMap()

        val business_profile = bid_wds_tf.map((record) => {
            val bid = record._1
            val word = record._2
            val tf_value = record._3
            val idf_val = idf_vals(word)
            (record._1, (word, tf_value * idf_val))
        }).groupByKey()
            .map((record) => {
                val bid_word_tfidf_list = record._2.toList.sortBy((value) => -1 * value._2).take(200)
                (index_to_bid_map(record._1), bid_word_tfidf_list)
            }).collect()
        val business_dict = business_profile.toMap

        val business_res = business_profile.map((record) => {
            val bid = record._1
            val words = record._2.map(str_dub_tup => str_dub_tup._1)
            Map("bid" -> bid, "words" -> words, "t" -> "b")
        })

        val user_profile = extracted_data_rdd.map((raw_data_tup) => (user_index_dict(raw_data_tup._1), raw_data_tup._2))
            .groupByKey()
            .mapValues((business_list) => {
                val business_distinct = business_list.toSet.toList
                business_distinct.flatMap((business) => {
                    business_dict(business)
                })
            })
            .map((record) => {
                val uid = index_to_uid_map(record._1)
                val lists = record._2
                val words = lists.take(500).map((record) => {
                    record._1
                })
                (uid, words)
            }).collect()
        val user_data = user_profile.map((record) => {
            val uid = record._1
            val words = record._2
            Map("uid" -> uid, "words" -> words, "t" -> "u")
        })

        val writer = new PrintWriter(new File(modelFilePath))
        business_res.foreach((record) => {
            implicit val formats = DefaultFormats
            writer.write(Serialization.write(record) + "\n")
        })
        user_data.foreach((record) => {
            implicit val formats = DefaultFormats
            writer.write(Serialization.write(record) + "\n")
        })
        writer.close()
    }
}