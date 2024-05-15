docker run -v ~/DE300/Lab6/spark-sql:/tmp/spark-sql -it \
           -p 8888:8888 \
           --name spark-sql-container \
	   pyspark-image
