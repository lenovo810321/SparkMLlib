package cn.spark.ml

import org.apache.spark.ml.feature.NGram
import org.apache.spark.sql.SparkSession

/**
  * Created by 张宝玉 on 2018/8/10.
  *
  * 一个n-gram是一个长度为整数n的字序列。NGram可以用来将输入转换为n-gram。
  */
object NGram {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master("local").appName("NGram").getOrCreate()
    val wordDataFrame = spark.createDataFrame(Seq(
      (0, Array("Hi", "I", "heard", "about", "Spark")),
      (1, Array("I", "wish", "Java", "could", "use", "case", "classes")),
      (2, Array("Logistic", "regression", "models", "are", "neat"))
    )).toDF("id", "words")

    val ngram = new NGram().setN(2).setInputCol("words").setOutputCol("ngrams")
    val ngramDataFrame = ngram.transform(wordDataFrame)
    ngramDataFrame.select("ngrams").show(false)

    spark.stop()
  }
}
