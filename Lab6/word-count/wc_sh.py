from pyspark import SparkContext, SparkConf

DATA = "./data/sh.txt"
OUTPUT_DIR = "keep3ormore_sh"

def filter_words():
 sc = SparkContext("local","Filter Words sh.txt")
 textFile = sc.textFile(DATA)
 filtered_words = textFile.flatMap(lambda line: line.split()) \
         .filter(lambda word: len(word) >= 3)
 filtered_words.saveAsTextFile(OUTPUT_DIR)
 # Split each line into words and flatten the result
 words = filtered_words.flatMap(lambda line: line.split())
 # Count the total number of words
 word_count = words.count()
 print("Number of words: ", word_count)

filter_words()
