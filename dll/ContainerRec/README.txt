On Ubuntu

1. install Python3.X
2. pip install -r requirements.txt
3. sudo apt install tesseract-ocr   (ubuntu 18.XX)
4. Remove all .traineddata files in /usr/share/tesseract-ocr/tessdata
5. copy "English.traineddata" to  /usr/share/tesseract-ocr/tessdata
6, Test: 
   $ python
   >>>import numrec
   >>>numrec.num_rec("./img/1.jpg")