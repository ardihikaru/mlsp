# Summary of performance evaluation

## Evaluation with all Dataset (Training=60k; Test=10k); Google Colab
| No. 	| Methods 	| #k 	| Accuracy (%) 	|
|:---:	|:---------------:	|:----:	|:-----------------:	|
| 1 	| kNN Scratch 	| 1 	| ???<br>(too long) 	|
| 2 	| kNN Sklearn 	| 1 	| 96.20 % 	|
| 3 	| kNN Sklearn (2) 	| 1000 	| 83.6 % 	|
| 4 	| kNN Theano GPU 	| 1000 	| 96.90 % 	|
| 5 	| Linear SVM 	| - 	| 92.00 % 	|
| 6 	| KDE 	| - 	| ???<br>(too long) 	|

## SVM
| No. 	| # Training 	| # Test 	| Accuracy (%) 	|
|:---:	|:----------:	|:------:	|:------------:	|
| 1 	| 80 	| 20 	| 0.60 % 	|
| 2 	| 800 	| 200 	| 0.88 % 	|
| 3 	| 2000 	| 200 	| 0.90 % 	|
| 4 	| 4000 	| 400 	| 0.87 % 	|
| 5 	| 10,000 	| 1,000 	| 0.86 % 	|

## Evaluation with Training=800 and Test=200
| No. 	| Methods 	| #k 	| Accuracy (%) 	| Longest Exec. Time (seconds) 	|
|:---:	|:--------------:	|:--:	|:------------:	|:----------------------------:	|
| 1 	| kNN Scratch 	| 50 	| 87.00 % 	| 0.94 s 	|
| 2 	| kNN Sklearn 	| 50 	| 85.50 % 	| 0.30 s 	|
| 3 	| kNN Theano CPU 	| 50 	| 71.00 % 	| 8.52 s 	|
| 4 	| kNN Theano GPU 	| 50 	| 71.00 % 	| 0.50 s 	|
| 5 	| Linear SVM 	| - 	| 88.00 % 	| 0.14 s 	|
| 6 	| KDE 	| - 	| 84.50 % 	| 0.99 s 	|


## KDE
| No. 	| # Training 	| # Test 	| Accuracy (%) 	| Bandwidth 	|
|:---:	|:----------:	|--------	|:------------:	|:---------:	|
| 1 	| 80 	| 20 	| 65.00 	| 2.0 	|
| 2 	| 800 	| 200 	| 84.50 	| 9.5 	|
