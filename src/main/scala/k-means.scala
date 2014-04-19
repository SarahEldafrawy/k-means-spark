/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import org.apache.spark.SparkConf

import java.util.Random
import org.apache.spark.SparkContext
import org.apache.spark.util.Vector
import org.apache.spark.SparkContext._
import org.apache.log4j.{ LogManager, Level }
import java.io._

/**
 * K-means clustering.
 */
object SparkKMeans {
  // LogManager.getRootLogger().setLevel(Level.WARN)

  def parseVector(line: String): Vector = {
    new Vector(line.split(' ').map(_.toDouble))
  }
  
  def findClosest(p: Vector, centers: Seq[Vector]): Int = {
    var index = 0
    var bestIndex = 0
    var closest = Double.PositiveInfinity
  
    for (i <- 0 until centers.length) {
      val tempDist = p.squaredDist(centers(i))
      if (tempDist < closest) {
        closest = tempDist
        bestIndex = i
      }
    }
  
    bestIndex
  }

  def average(points: Seq[Vector]) : Vector = {
    points.reduce(_+_) / points.length
  }


  def main(args: Array[String]) {
    if (args.length < 5) {
        println(args.length)
        System.err.println("Usage: SparkLocalKMeans <master> <file> <k> <convergeDist> <output file> <number of simulations> <number of cores>")
        System.exit(1)
    }
    // Get constants from command line
    // 1) local | local[k] | spark://HOST:PORT | mesos://HOST:PORT
    val sparkHost = args(0)
    // 2) input filename, relative to project directory, e.g. myInputData.txt
    val inputName = args(1)
    // 3) number of clusters
    val K = args(2).toInt
    // 4) max change of the center of the clusters between iterations before the clustering stops
    val convergeDist = args(3).toDouble
    val outputFile = args(4)
    // 5) number of simulations
    val numSimulations = args(5).toInt

    val conf = new SparkConf()
             .setMaster(sparkHost)
             .setAppName("kMeansRecur")
             .set("spark.cores.max", args(6))
    val sc = new SparkContext(conf)

    // Make Spark Context
    // sc = new SparkContext(, , System.getenv("SPARK_HOME"), SparkContext.jarOfClass(this.getClass))

    // Make an RDD (Resilient Distributed Dataset = basic data set to work with in Spark) of given textfile as first argument
    val textFile = sc.textFile(inputName)

    // Get a list of points (point = vector of "coordinates") from the input data
    // Line input format: (x y z ...)
    val points = textFile.map( line => new Vector(line.split(' ').map(_.toDouble)))

    // Cache the points in memory of the clusters, because we are going to need it a lot.
    points.cache()

    /* 
     * The actual k-means algorithm
     * Based on pseudo code: http://en.wikibooks.org/wiki/Data_Mining_Algorithms_In_R/Clustering/K-Means
     */

    var benchmarks : Seq[Long] = Seq()
    for (i <- (1 until numSimulations)) {
      var dist = sc.accumulator(1.0)

      // 1) Take k initial points as clusters
      var centroids = points.takeSample(false, K, scala.util.Random.nextInt)


       // 2) start iterations
      val start = System.nanoTime
      // while(dist.value > convergeDist) {
      for (j <- (i until 100)) {
        // for every point, find the closest centroid
        val closest = points.map( point => (findClosest(point, centroids), point))

        // accumulator for differences new centroids
        dist = sc.accumulator(0.0)

        // calculate new centroids + add difference to old centroids
        centroids = closest.groupByKey().map{case(i, points) =>
          val newCentroid = average(points)
          dist += centroids(i).squaredDist(newCentroid)
          newCentroid
        }.collect()
    
        // println("Finished iteration (delta = " + tempDist + ")")
      }

      // println("Final centers:")
      // kPoints.foreach(println)


      val micros = (System.nanoTime - start) / 1000
      benchmarks = micros +: benchmarks
      println("%d: %d".format(i, micros))
    }
    

    benchmarks.foreach( micro => println(micro))
    println("\n => %d".format(benchmarks.reduce(_+_)/benchmarks.length))
    val pw = new PrintWriter(outputFile)
    benchmarks.foreach(pw.println(_))
    pw.close
    
    // Print results
    // val finalCentroids = result._1
    // val dist = result._2
    // println("Final centers:")
    // finalCentroids.foreach(println)
    // println("Clusters:")
    // points.toArray.foreach(point => println(point + " => " + (finalCentroids(findClosest(point, finalCentroids)))))
    System.exit(0)
  }
}
