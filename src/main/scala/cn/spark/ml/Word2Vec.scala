package cn.spark.ml

import org.apache.spark.ml.feature.Word2Vec
import org.apache.spark.sql.SparkSession

/**
  * Created by 张宝玉 on 2018/8/8.
  *
  * 4.1.2 Word2Vec
  *
  * Word2vec是一个Estimator，它采用一系列代表文档的词语来训练word2vecmodel。
  * 在下面的代码段中，我们首先用一组文档，其中每一个文档代表一个词语序列。
  * 对于每一个文档，我们将其转换为一个特征向量(单词->数值向量)。
  * 此特征向量可以被传递到一个学习算法。
  */
object Word2Vec {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().appName("Word2Vec").master("local").getOrCreate()
    // 输入数据，每行为一个词袋，可来自语句或文档。
    val documentDF = spark.createDataFrame(Seq(
      "Hi I heard about Spark".split(" "),
      "I wish Java could use case classes".split(" "),
      "Logistic regression models are neat".split(" ")
    ).map(Tuple1.apply)).toDF("text")

    //训练从词到向量的映射
    val word2Vec = new Word2Vec()
      .setInputCol("text")
      .setOutputCol("result")
      .setVectorSize(3)
      .setMinCount(0)
    val model = word2Vec.fit(documentDF)
    val result = model.transform(documentDF)
    result.select("result").take(3).foreach(println)
  }
}
