package cn.spark.ml

import org.apache.spark.ml.feature.{PCA, StandardScaler}
import org.apache.spark.sql.SparkSession

/**
  * Created by 张宝玉 on 2018/8/10.
  *
  * 注意：PCA前一定要对特征向量进行规范化（标准化）！！！
  */
object PCA {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder().master("local").appName("").getOrCreate()

    val rawDataFrame = spark.read.format("libsvm").load("F:\\src\\SparkMLlib\\data\\transform\\sample_libsvm_data.txt")

    val scaledDataFrame = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("scaledFeatures")
      .setWithMean(false)
      .setWithStd(true)
      .fit(rawDataFrame)
      .transform(rawDataFrame)

    val pcaModel = new PCA().setInputCol("scaledFeatures")
      .setOutputCol("pcaFeatures")
      .setK(3)
      .fit(scaledDataFrame)

    pcaModel.transform(scaledDataFrame).select("label", "pcaFeatures").show(10,false)

    //如何选择k值？
    val pcaModel2 = new PCA().setInputCol("scaledFeatures")
      .setOutputCol("pcaFeatures2")
      .setK(50)//
      .fit(scaledDataFrame)
    var i = 1
    for(x<-pcaModel2.explainedVariance.toArray){
      println(i+"\t"+x+" ")
      i += 1
    }
    spark.stop()
  }
}

//+-----+-------------------------------------------------------------+
//|label|pcaFeatures                                                  |
//+-----+-------------------------------------------------------------+
//|0.0  |[-14.998868464839624,-10.137788261664621,-3.042873539670117] |
//|1.0  |[2.1965800525589754,-4.139257418439533,-11.386135042845101]  |
//|1.0  |[1.0254645688925883,-0.8905813756164163,7.168759904518129]   |
//|1.0  |[1.5069317554093433,-0.7289177578028571,5.23152743564543]    |
//|1.0  |[1.6938250375084654,-0.4350617717494331,4.770263568537382]   |
//|0.0  |[-15.870371979062549,-9.999445137658528,-6.521920373215663]  |
//|1.0  |[2.211273240523006,-3.9211298765685063,-11.142217103787143]  |
//|1.0  |[-2.8590273038441603,-5.612660590695614,-2.057549382835385]  |
//|0.0  |[-9.992352316606867,-16.856375320019165,0.31303581897017874] |
//|0.0  |[-13.097863259806243,-17.058378993308917,-0.9846688822878218]|
//+-----+-------------------------------------------------------------+