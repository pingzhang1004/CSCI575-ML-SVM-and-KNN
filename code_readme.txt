Name: Ping Zhang
CSM ID:10909957

Programming language: Python

How the code is structured: 
 - First, we defined a couple of key parameters (e.g., color table for creating final land-cover map)
 - Second, step into the main function, import image data and split it into training and testing samples (run PCA if needed)
 - Third, run KNN or SVM classification model, Please note that the implementation of KNN and SVM are located out of the main function.
 - Fourth, generate and save qualitative and quantitative results.

How to run the code:
 - Setup Python environment (using either Miniconda or Anaconda)
   * Python = 3.9
   * numpy = 1.26.1
   * scipy = 1.11.3
   * scikit-learn = 1.3.2
   * imageio = 2.32.0
   * pickle
   * random
 - For the specific Python script that you want to run, modify corresponding variables, including input and output file paths.
 - Run Python script
   * python "python_file_name.py"
