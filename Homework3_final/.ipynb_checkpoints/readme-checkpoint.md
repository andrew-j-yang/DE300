Instructions to run

1) go to AWS EMR, and clone the cluster titled 'andrewsparkcluster'
2) uncheck the option 'Include 1 step(s) with new cluster' and edit configurations
3) leave all settings the same, and click 'clone cluster'
4) click 'add step'
5) select type as 'Spark application'
6) type the application location as 's3://andrewawsbucket/homework3_andrew_latest.py'
7) type the spark-submit options as '--master yarn'
8) select step action as 'cancel and wait'