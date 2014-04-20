import org.apache.spark.SparkConf

import java.util.Random
import org.apache.spark.SparkContext
import org.apache.spark.util.Vector
import org.apache.spark.SparkContext._
import org.apache.log4j.{ LogManager, Level }
import java.io._

object KMeans {
  // LogManager.getRootLogger().setLevel(Level.WARN)

  // convert a string of type: "0.00 0.00 0.00 ..." to a vector of doubles
  def lineToDoubles(line: String): Vector = {
    new Vector(line.split(' ').map(_.toDouble))
  }

  def average(points: Seq[Vector]) : Vector = {
    points.reduce(_+_) / points.length
  }
  
  // Return the index of the closest centroid to given point.
  // Calculated by finding minimum Euclidean Distance.
  def closestCentroid(point: Vector, centroids: Seq[Vector]): Int = {
    var index = 0
    var bestIndex = 0
    var closest = Double.PositiveInfinity
  
    for (i <- 0 until centroids.length) {
      val tempDist = point.squaredDist(centroids(i))
      if (tempDist < closest) {
        closest = tempDist
        bestIndex = i
      }
    }
  
    bestIndex
  }


  def main(args: Array[String]) {
    if (args.length < 7) {
        println(args.length)
        System.err.println("Usage: sbt \"run <master> <file> <k> <convergeDist> <output file> <number of simulations> <number of cores>\"")
        System.exit(1)
    }

    /* 
     * Preparation
     */

    // 1) local | local[k] | spark://HOST:PORT | mesos://HOST:PORT
    // locak[k] => run locally with k workers
    val sparkHost = args(0)
    // 2) input filename, relative to project directory, e.g. data/myInputData.txt
    val inputName = args(1)
    // 3) number of k-clusters
    val K = args(2).toInt
    // 4) max change of the center of the clusters between iterations before the clustering stops
    val convergeDist = args(3).toDouble
    // 5) file to write output to
    val outputFile = args(4)
    // 6) number of simulations
    val numSimulations = args(5).toInt
    // 7) number of cores for each worker
    val numCores = args(6)


    // create spark context
    val conf = new SparkConf()
             .setMaster(sparkHost)
             .setAppName("KMeans")
             .set("spark.cores.max", numCores)
    val sc = new SparkContext(conf)

    // Make an RDD (Resilient Distributed Dataset = basic data set to work with in Spark) of given inputfile
    val textFile = sc.textFile(inputName)

    // Get a list of points (point = vector of "coordinates") from the input data
    val points = textFile.map(lineToDoubles _)

    // Cache the points in memory of the clusters!
    points.cache()


    
    /* perform the algorithm a given number of times to average out benchmarks */
    var benchmarks : Seq[Long] = Seq()
    for (i <- (1 until numSimulations)) {


      /* 
       * The actual k-means algorithm
       * Based on pseudo code: http://en.wikibooks.org/wiki/Data_Mining_Algorithms_In_R/Clustering/K-Means
       */

      // * start benchmarking
      val start = System.nanoTime
      
      // accumulator that will be used to compute the total distance that all centroids moved in this iteration
      var dist = sc.accumulator(1.0)

      // 1) Take k random initial points as centroids
      var centroids = points.takeSample(false, K, scala.util.Random.nextInt)


      // 2) start iterations
      while(dist.value > convergeDist) {
        // reset accumulator
        dist = sc.accumulator(0.0)

        // for every point, find the closest centroid
        val closest = points.map (point => (closestCentroid(point, centroids), point))

        // calculate new centroids + add difference to old centroids
        centroids = closest.groupByKey().map {case(i, points) =>
          val newCentroid = average(points)
          dist += centroids(i).squaredDist(newCentroid)
          newCentroid
        }.collect()
    
        println("delta = " + dist.value)
      }


      // * stop benchmarking
      val micros = (System.nanoTime - start) / 1000
      benchmarks = micros +: benchmarks
      println("%d: %d".format(i, micros))
    }

    // print benchmarks
    benchmarks = benchmarks.reverse
    benchmarks.foreach( micro => println(micro))
    println("\n => %d".format(benchmarks.reduce(_+_)/benchmarks.length))

    // write to output file
    val pw = new PrintWriter(outputFile)
    benchmarks.foreach(pw.println(_))
    pw.close

    System.exit(0)
  }
}
