import breeze.numerics.sqrt
import scala.io.Source
import scala.math.log
import scala.io.Source
import scala.util.Random
import scala.collection.mutable.ListBuffer


import org.apache.log4j.{Level, Logger}

case class Point(features: List[Double], label: Int)

object HeartDisease {

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.OFF)
    Logger.getLogger("akka").setLevel(Level.OFF)

    val rdd = Source.fromFile("heart_disease.csv").getLines.toList

    val n = 1000

    val Y = rdd
      .map(_.split(",")(0))
      .map(_.toInt)
      .take(n)

    val X: List[List[Double]] = rdd
      .map(_.split(",", 2)(1))
      .map(_.split(",").map(_.toDouble).toList)
      .take(n)

    // Split data into training and testing sets
    val splitRatio = 0.8
    val splitIndex = (X.length * splitRatio).toInt
    val (trainX, testX) = X.splitAt(splitIndex)
    val (trainY, testY) = Y.splitAt(splitIndex)

    // Run KNN
    val maxK = 30 //change to test different values of k
    val kValues = (1 to maxK).toList

    val result = kValues.map{ k =>
      val preds = knn(trainX, trainY, k, testX)
      val macroF1 = macroF1Score(preds, testY)
      (k, macroF1)
    }.sortBy(-_._2)

    result.foreach(println)
  }

  def euclideanDistance(vector1: List[Double], vector2: List[Double]): Double = {
    require(vector1.length == vector2.length, "Vectors must have the same length")

    val squaredDistances = vector1.zip(vector2).map { case (x1, x2) =>
      val diff = x1 - x2
      diff * diff
    }

    math.sqrt(squaredDistances.sum)
  }

  def knn(X: List[List[Double]], Y: List[Int], k: Int, points: List[List[Double]]): List[Int] = {
    points.map { newPoint =>
      val allPoints = X.zip(Y).map { case (features, label) => Point(features, label) }

      val nearestNeighbors = allPoints.map { p =>
        (p, euclideanDistance(p.features, newPoint))
      }.sortBy(_._2).take(k)

      val labelCounts = nearestNeighbors.map(_._1.label).groupBy(identity).mapValues(_.size)
      labelCounts.maxBy(_._2)._1
    }
  }

  def macroF1Score(predictions: List[Int], labels: List[Int]): Double = {
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

}
