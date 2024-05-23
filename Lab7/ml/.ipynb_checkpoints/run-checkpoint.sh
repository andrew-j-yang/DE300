docker run -v ~/DE300/Lab7/ml:/tmp/ml -it \
           -p 8888:8888 \
           --name spark-sql-container \
	   pyspark-image
