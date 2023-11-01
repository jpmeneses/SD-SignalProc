# Shoulder Exercise: Classification and Velocity Loss

This private repository will be used for our project on the shoulder exercise dataset.

Don't hesitate to work together and exchange idea.


Keep the "Data experiment" and "Data" folder one level above the project 


# Let's describe which code is doing what:

Classes folder:

  - Plotter.py: use to plot and click on the data to cut
  - Path.py: use to get all the variables related to path ...
  - Data.py: user-dependent cv; validation and test set are from same participant 
  - Data_2.py:  usr-independent nested K-fold cv;  6 outer folds, 5 inner folds 
  
  - Segmentation.py: segmentation function for reps/ecc/conc

  - Model.py: different neural network models
  
 Main folder:
 
  - PointData.py: code use to get the reps/ecc/conc points
  - PointData2.py:  code use to subtitute the rep/conc/ecc by manually seg to the old one.
  - Sliding window.py: code use to cut the data by sliding window for machine learning
  
  - Main_ML_1.py:  subject-dependent Conventionnal Machine Learning (mostly with sklearn)
  - Main_ML_2.py:  subject-independent Conventionnal Machine Learning (mostly with sklearn)
  
  - AngularToLinear.py: code use to change angular velocity to linear velocity
  - Con_Ecc_Rep: code use to run class of Segmentation for rep/ecc/conc vel segmentation
  - GatherMarker.py: code use to gather motion capture data and compute length of upper/lower arm
  - Load_eccconc_data.py:  code use to load all velocity data files
  - PointData2.py: code use to subtitute the rep/conc/ecc by manually seg to the old one.


# Usage

- Environment

    - Python 3.9

    - OpenCV, scikit-image, tqdm, oyaml

    - *we recommend [Anaconda](https://www.anaconda.com/distribution/#download-section) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html#linux-installers), then you can create the TensorFlow 2.2 environment with commands below*

        ```console
        conda create -n sd python=3.9

        conda install -c anaconda ipykernel scikit-learn pandas seaborn git

        conda install -c conda-forge matplotlib tabulate notebook jupyterlab openjdk

        python -m pip install "git+https://github.com/digitalinteraction/openmovement-python.git#egg=openmovement"

        python -m pip install accelerometer
        ```

    - *NOTICE: if you create a new conda environment, remember to activate it before any other command*

        ```console
        source activate sd
        ```