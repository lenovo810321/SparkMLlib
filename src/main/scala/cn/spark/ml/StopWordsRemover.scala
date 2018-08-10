package cn.spark.ml

import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.sql.SparkSession

/**
  * Created by 张宝玉 on 2018/8/10.
  *
  * 停用词为在文档中频繁出现，但未承载太多意义的词语，
  * 它们不应该被包含在算法输入中，所以会用到移除停用词（StopWordsRemover）
  */
object StopWordsRemover {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master("local").appName("StopWordsRemover").getOrCreate()
    val remover = new StopWordsRemover()
      .setInputCol("raw")
      .setOutputCol("filtered")

    val dataSet = spark.createDataFrame(
      Seq((0, Seq("I", "saw", "the", "red", "balloon")),
        (1, Seq("Mary", "had", "a", "little", "lamb")))
    ).toDF("id", "raw")
    remover.transform(dataSet).show(false)
  }
}
