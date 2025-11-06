package org.rubigdata

import org.apache.spark.sql.functions.lit
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.scalalang.typed
import org.apache.spark.sql.functions.udf

import com.johnsnowlabs.nlp._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.annotator._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.StopWordsRemover
import org.apache.spark.sql.functions._

import org.apache.spark.graphx._
import org.apache.spark.rdd.RDD
import org.apache.spark.graphx.lib.ShortestPaths
import org.apache.spark.graphx.lib.ConnectedComponents
import scala.collection.mutable.Queue

import org.jsoup.Jsoup
import org.jsoup.nodes.Document
import org.apache.spark.sql.functions.udf

import scala.math.log



object RUBigDataApp {
	def main(args: Array[String]) {
		val spark = SparkSession.builder.appName("RUBigDataProject").getOrCreate()
		import spark.implicits._
		spark.sparkContext.setLogLevel("WARN")

// ================================ READING WARC FILES AND CONVERTING TO TEXT ================================

		val warcResponses = spark
		    .read
		    .format("org.rubigdata.warc")
		    .load("hdfs:///single-warc-segment/")
		    .filter($"warcType" === "response")
		    .select($"warcTargetUri", $"warcBody")
		    .toDF()

		// UDF to clean HTML
		val extractText = udf((httpResponse: String) => {
		    if (httpResponse == null || httpResponse.isEmpty) null
		    else {
		        try {
		            // Split on first empty line (i.e. \r\n\r\n)
		            val htmlStart = httpResponse.split("\r\n\r\n", 2)
		            if (htmlStart.length < 2) null // no HTML found
		
		            val html = htmlStart(1)
		            val doc = Jsoup.parse(html)
		            val lang = Option(doc.select("html").first().attr("lang")).getOrElse("").toLowerCase
		            if (!lang.startsWith("en")) null
		            else {
		                doc.select("script, style, noscript").remove()
		                doc.body().text()
		            }
		        } catch {
		            case _: Throwable => null
		        }
		    }
		})
		
		// Apply UDF
		val WETData = warcResponses.select("warcBody").withColumn("text", extractText($"warcBody")).filter(col("text").isNotNull)
		

// ================================ PARSE TEXT ================================
		// NLP PIPELINE
		// Document Assembler
		val doc = new DocumentAssembler()
			.setInputCol("text")
			.setOutputCol("document")

		// Tokenizer
		val tokenizer = new Tokenizer()
			.setInputCols("document")
			.setOutputCol("token")

		// Normalizer - removing non-alpha character
		val normalizer = new Normalizer()
			.setInputCols("token")
			.setOutputCol("normalized")
			.setLowercase(true)
			.setCleanupPatterns(Array("[^a-zA-Z]"))

		// StopWordCleaner
		// Filter short words
		val shortWords = ('a' to 'z').map(_.toString) ++
		    (for {
		        a <- 'a' to 'z'
		        b <- 'a' to 'z'
		    } yield s"$a$b")
		val defaultStopWords = StopWordsRemover.loadDefaultStopWords("english")
		val customStopWords = defaultStopWords ++ shortWords

		val stopWordCleaner = new StopWordsCleaner()
		    .setInputCols("normalized")
		    .setOutputCol("filtered")
		    .setCaseSensitive(false)
		    .setStopWords(customStopWords.distinct)
            

		// POS Tagging
		val posTagger = PerceptronModel.pretrained("pos_anc", "en")
			.setInputCols("document", "filtered")
			.setOutputCol("pos")

		// Lemmatizer
		val lemmatizer = LemmatizerModel.pretrained("lemma_antbnc", "en")
			.setInputCols("filtered")
			.setOutputCol("lemma")

		// Finisher
		val finisher = new Finisher()
			.setInputCols("lemma", "pos")
			.setIncludeMetadata(true)

		// Pipeline
		val pipeline = new Pipeline().setStages(Array(
				doc, tokenizer, normalizer, stopWordCleaner, posTagger, lemmatizer, finisher
			))

		// spedup fit
		val dummy = spark.createDataset(Seq("")).toDF("text")
		
		val model = pipeline.fit(dummy)
		
		// val LIMIT = 10
		// val result = model.transform(WETData.limit(LIMIT))
		val result = model.transform(WETData)

  		// val model = pipeline.fit(WETData)
		// val result = model.transform(WETData)

		val filterNouns = udf {
		    (lemmas: Seq[String], pos: Seq[String]) => 
		    (lemmas zip pos).filter(_._2.startsWith("NN")).map(_._1)
		}
		
		val filtered_result = result.withColumn("nouns", filterNouns(
		    col("finished_lemma"),
		    col("finished_pos")
		))

// ================================ CONSTRUCTING THE GRAPH ================================
		
		val nounsToTopics = udf {
		    nouns: Seq[String] => 
		    nouns.groupBy(identity)
		        .mapValues(_.size)
		        .toSeq
		        .sortBy(-_._2)
		        .take(5)
		        .map(_._1)
		}
		
		val topicData = filtered_result.withColumn("topics", nounsToTopics(col("nouns")))

		
		val explode1 = topicData
		    .withColumn("topic1", explode(col("topics")))
		    .withColumn("topics1", col("topics"))
		    .select("topics1", "topic1")
		  
		 val explode2 = topicData
		    .withColumn("topic2", explode(col("topics")))
		    .withColumn("topics2", col("topics"))
		    .select("topics2", "topic2")
		  
		 val associationGraph = explode1
		    .join(explode2,  ($"topics1" === $"topics2") && ($"topic1" < $"topic2"))
		    .select("topic1", "topic2")
		    .groupBy("topic1", "topic2")
		    .count()

		val topic_occurence = topicData
		    .withColumn("exploded topics", explode(col("topics")))
		    .groupBy("exploded topics")
		    .count()
		    .select("exploded topics", "count")

		val A_occur = topic_occurence.withColumnRenamed("count", "#A").withColumnRenamed("exploded topics", "A")
		val B_occur = topic_occurence.withColumnRenamed("count", "#B").withColumnRenamed("exploded topics", "B")
		
		val N = topic_occurence.agg(sum("count")).first().getLong(0).toDouble
		
		
		val associationGraphWithCounts = associationGraph
			.withColumnRenamed("count", "#AB")
			.join(A_occur, $"topic1" === $"A")
			.join(B_occur, $"topic2" === $"B")
			.withColumn("N", lit(N))
			.select("topic1", "topic2", "#AB", "#A", "#B", "N")
		
		val PMI = udf { (AB: Int, A: Int, B: Int, N: Int) =>
		    math.log((AB.toDouble * N) / (A.toDouble * B.toDouble)) / math.log(2)
		}
		
		val associationGraphWithPMI_ = associationGraphWithCounts
		    .withColumn("PMI", PMI(col("#AB"), col("#A"), col("#B"), col("N")))
		    .select("topic1", "topic2", "PMI")
		
		val stats = associationGraphWithPMI_.agg(
		  mean("PMI").alias("mean"),
		  stddev("PMI").alias("stddev")
		).collect()(0)
		
		val meanValue = stats.getAs[Double]("mean")
		val stddevValue = stats.getAs[Double]("stddev")
		
		val associationGraphWithPMI = associationGraphWithPMI_.filter(col("PMI") >= meanValue - 1.96 * stddevValue)
	
		// get all the vertices
		val vertices : RDD[(VertexId, String)] = associationGraphWithPMI
			.select("topic1")
			.union(associationGraphWithPMI.select("topic2"))
			.distinct()
			.rdd.map(_.getAs[String](0))    // we convert to RDD which is required by GraphX
			.zipWithIndex                   // GraphX works with longs not strings, so we zip with index to get a UID for each word
			.map(_.swap)

		// Edge RDD
		val map = vertices.map { case (id, name) => (name, id) }.collectAsMap() // Map[String, Long]
		val broadcastMap = spark.sparkContext.broadcast(map)
		
		val edges = associationGraphWithPMI.rdd.flatMap { row =>
		  val topic1 = row.getAs[String]("topic1")
		  val topic2 = row.getAs[String]("topic2")
		  val pmi = row.getAs[Double]("PMI")
		
		  val map = broadcastMap.value
		  for {
		    id1 <- map.get(topic1)
		    id2 <- map.get(topic2)
		  } yield Edge(id1, id2, pmi)
		}
		val dir_graph = Graph(vertices, edges)
		val graph = Graph(dir_graph.vertices, dir_graph.edges.union(dir_graph.edges.map(e => Edge(e.dstId, e.srcId, e.attr))))
		
// ================================ GRAPH ANALYSIS ================================
	
		// Simple Graph stats 
		val n_vertices = graph.vertices.count()
		val n_edges = graph.edges.count()

        val wordToId: Map[String, VertexId] = vertices.map(_.swap).collectAsMap().toMap
		val idToWord: Map[VertexId, String] = vertices.collectAsMap().toMap

		// Graph hotspots
		val degrees = graph.degrees
			.sortBy(_._2, false)
			.map {case (id, degree) => (idToWord(id), degree)}
            .take(10)
			.toSeq
			.toDF("name", "degrees")

        // ConnectedComponents
		val ccGraph = ConnectedComponents.run(graph)
			.vertices
			.map { case (_, componentId) => (componentId, 1) }
			.reduceByKey(_ + _)
			.filter(_._2 > 5)
			.map { case (vertex, cId) => (idToWord(vertex), cId) }
			.sortBy(_._2, ascending = false)
            .take(20)
			.toSeq
			.toDF("name", "size")

        // Word furthest from "online"
		val shortestPathsToOnline = ShortestPaths
			.run(graph, Seq(wordToId("online")))
			.vertices
			.filter {case (_, path) => path.contains(wordToId("online"))}
			.map {case (id, path) => (id, path(wordToId("online")))}
			.sortBy(_._2, false)
			.take(10)
			.map {case (id, length) => (idToWord(id), length)}
			.toSeq
			.toDF("word", "distance")

        // Least associated words
		var prev_dist = -1
		var cur_dist = 0
		var prev_word = wordToId("online")
		var cur_word = wordToId("online")
		var iter = 0
		val MAX_ITER = 10000

		while (prev_dist < cur_dist && iter < MAX_ITER) {
			val shortestPathsGraph = ShortestPaths.run(graph, Seq(cur_word))

			val next = shortestPathsGraph.vertices
				.filter { case (_, path) => path.contains(cur_word) }
				.map { case (id, path) => (id, path(cur_word)) }
				.max()(Ordering.by(_._2))

			prev_word = cur_word
			prev_dist = cur_dist

			cur_word = next._1
			cur_dist = next._2

			iter += 1
		}

		val furthest_word_1 = idToWord(prev_word)
		val furthest_word_2 = idToWord(cur_word)
		val furthest_distance = cur_dist

	   
// ================================ SAVE DATA ================================

		val valueResults = Seq(
			("n_vertices", n_vertices.toString),
			("n_edges", n_edges.toString),
            ("furthest_word_1", furthest_word_1),
			("furthest_word_2", furthest_word_2),
			("furthest_distance", furthest_distance.toString)
			)
			.toDF("metric", "value")

		valueResults
			.write
			.mode("overwrite")
			.format("parquet")
			.save("hdfs:///user/s1098110/output_simple_graph_stats")

        degrees
			.write
			.mode("overwrite")
			.format("parquet")
			.save("hdfs:///user/s1098110/output_degrees")

        ccGraph
			.write
			.mode("overwrite")
			.format("parquet")
			.save("hdfs:///user/s1098110/output_cc")

        shortestPathsToOnline
			.write
			.mode("overwrite")
			.format("parquet")
			.save("hdfs:///user/s1098110/output_shortest_path_to_online")


		spark.stop()
	}
}
