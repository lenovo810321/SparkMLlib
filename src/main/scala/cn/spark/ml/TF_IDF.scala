package cn.spark.ml

import org.apache.spark.ml.feature.{HashingTF, IDF, Tokenizer}
import org.apache.spark.sql.SparkSession

/**
  * Created by 张宝玉 on 2018/8/8.
  *
  * 4.1.1 词频－逆向文件频率（TF-IDF）
  *
  * 词频－逆向文件频率（TF-IDF）是一种在文本挖掘中广泛使用的特征向量化方法，
  * 它可以体现一个文档中词语在语料库中的重要程度。
  * 在下面的代码段中，我们以一组句子开始。首先使用分解器Tokenizer把句子划分为单个词语。
  * 对每一个句子（词袋），我们使用HashingTF将句子转换为特征向量，最后使用IDF重新调整特征向量。
  * 这种转换通常可以提高使用文本特征的性能。
  */
object TF_IDF {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master("local").appName("TF_IDF").getOrCreate()
    val sentenceData = spark.createDataFrame(Seq(
      (0, "Hi I heard about Spark"),
      (0, "I wish Java could use case classes"),
      (1, "Logistic regression models are neat")
    )).toDF("label", "sentence")

    val tokenizer = new Tokenizer().setInputCol("sentence").setOutputCol("words")
    val wordsData = tokenizer.transform(sentenceData)
    val hashingTF = new HashingTF().setInputCol("words").setOutputCol("rawFeatures").setNumFeatures(20)
    val featurizedData = hashingTF.transform(wordsData)
    // CountVectorizer也可获取词频向量

    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    val idfModel = idf.fit(featurizedData)
    val rescaledData = idfModel.transform(featurizedData)
    rescaledData.select("features", "label").take(3).foreach(println)
  }
}
