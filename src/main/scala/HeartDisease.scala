package example
import scala.io.Source
import org.apache.log4j.{Level, Logger}
import org.apache.spark.{ SparkConf, SparkContext }
import org.apache.spark.rdd._


case class Point(features: List[Double], label: Double)

object HeartDisease {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val conf = new SparkConf().setAppName("HeartDisease").setMaster("local[4]")
    val sc = new SparkContext(conf)

    val rdd = sc.parallelize(Source.fromFile("heart_disease.csv").getLines.toList)

    // Extracting the first column (Y)
    val Y: RDD[Double] = rdd
      .map(_.split(",")(0))
      .map(_.toDouble)

    // Extracting the rest of the columns (X)
    val X: RDD[List[Double]] = rdd
      .map(_.split(",", 2)(1))
      .map(_.split(",").map(_.toDouble).toList)

    // Split data into training and testing sets
    val splitRatio = 0.99999
    val splitRatioArray = Array(splitRatio, 1 - splitRatio)
    val seed = 123
    val Array(trainX, testX) = X.randomSplit(splitRatioArray, seed)
    val Array(trainY, testY) = Y.randomSplit(splitRatioArray, seed)

    // Run KNN
    println("Running KNN")
    val predictions = knn(trainX, trainY, 14, testX)
    println("KNN Complete")

    val predictionsList = predictions.collect().toList
    val testYList = testY.collect().toList

    val macroF1 = macroF1Score(predictionsList, testYList)
    println(macroF1)

    val confusionMatrixResult = confusionMatrix(predictionsList, testYList)
    println("Confusion Matrix:")
    confusionMatrixResult.foreach(row => println(row.mkString("\t")))

    sc.stop()
  }



  def knn(X: RDD[List[Double]], Y: RDD[Double], k: Int, points: RDD[List[Double]]): RDD[Double] = {
    val dataPoints = X.zip(Y).map { case (features, label) => Point(features, label) }
    val distances = points.cartesian(dataPoints).map { case (point, dataPoint) =>
      val distance = math.sqrt(point.zip(dataPoint.features).map { case (a, b) => math.pow(a - b, 2) }.sum)
      (point, (dataPoint.label, distance))
    }
    val kNearestNeighbors = distances.groupByKey().mapValues(_.toList.sortBy(_._2).take(k))
    val predictions = kNearestNeighbors.mapValues { neighbors =>
      neighbors.groupBy(_._1).mapValues(_.size).maxBy(_._2)._1
    }
    predictions.map(_._2)
  }


  def macroF1Score(predictions: List[Double], labels: List[Double]): Double = {
    require(predictions.length == labels.length, "Number of predictions must be equal to the number of labels")

    val uniqueLabels = (predictions ++ labels).distinct
    val f1Scores = uniqueLabels.map { label =>
      val truePositives = predictions.zip(labels).count { case (predicted, actual) =>
        predicted == label && actual == label
      }
      val falsePositives = predictions.zip(labels).count { case (predicted, actual) =>
        predicted == label && actual != label
      }
      val falseNegatives = predictions.zip(labels).count { case (predicted, actual) =>
        predicted != label && actual == label
      }
      val precision = truePositives.toDouble / (truePositives + falsePositives)
      val recall = truePositives.toDouble / (truePositives + falseNegatives)
      val f1 = 2 * (precision * recall) / (precision + recall)
      if (f1.isNaN) 0.0 else f1
    }

    f1Scores.sum / f1Scores.length
  }


  def confusionMatrix(predictions: List[Double], testY: List[Double]): Array[Array[Int]] = {
    val tp = predictions.zip(testY).count { case (pred, actual) => pred == 1.0 && actual == 1.0 }
    val fp = predictions.zip(testY).count { case (pred, actual) => pred == 1.0 && actual == 0.0 }
    val tn = predictions.zip(testY).count { case (pred, actual) => pred == 0.0 && actual == 0.0 }
    val fn = predictions.zip(testY).count { case (pred, actual) => pred == 0.0 && actual == 1.0 }

    Array(Array(tp, fp), Array(fn, tn))
  }

}
