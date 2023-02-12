# Thesis: Analysis of distinctive traits in the Parkinsonian handwriting 2021/2022

## Candidate

Vincenzo Salvati - v.salvati10@studenti.unisa.it - 0622701550

# Abstract

The aim of this thesis concerns the study of distinctive traits of handwriting motor control that define the behaviour of people affected by Parkinson’s disease. The search for these distinctive traits is addressed by using Machine Learning as the core technology to develop systems to support medical assessments in the diagnosis of the disease.
In the literature, different methodologies and tools have been proposed to deal with this problem, many of which are based on the use of a wide range of handwriting tasks, features, and classifiers. In such a context, the work reported in this thesis focuses on exploiting feature selection techniques to improve performance, but also to ascertain to which extent different tasks contribute with relevant features, with the ultimate aim of selecting the combinations of tasks, features and classifiers that lead to better performance. For this purpose, a large set of experiments has been performed on the PaHaw dataset, one of the most widely used in the literature, considering different feature selection techniques belonging to different categories, as well as different classifiers among those that have proved to be the best performing for the problem at hand. 
The experimental results, supported by robust statistical analysis, show significant improvements when feature selection is adopted with respect to a baseline system using all the features available in the dataset, going from about 65% to about 90% of accuracy. Furthermore, it has been possible to identify tasks that give more satisfaction than others when individually used to train Machine Learning models. Indeed, a ranking of the best configuration (model/feature selection technique) for every task has been established and each of them is defined as the expert of the related task. This ranking is used to develop an Ensemble Learning algorithm that aggregates the experts’ decisions by a majority voting approach. From this last experiment has been found a trade-off between performance and the number of tasks to use and what turns out is the non-feasible result, comparing the baseline’s best one. Thus, despite having found evidence that assesses the misleading of Machine Learning models by using some tasks, is not possible to reduce the number of these latter or, at least, is not possible by using this rule of aggregation or by using these feature selection techniques.

## Paths

```.
|-- diagnosticParkinsonDisease
|   |-- experiments
|   |   |-- baseline_experiment.py
|   |   |-- baseline_experiment_best_hyperparameters.py
|   |   |-- tasks_experiment.py
|   |   |-- tasks_experiment_best_hyperparameters.py
|   |   |-- tasks_final_ensemble_learning_experiment.py
|   |   |-- tasks_final_ensemble_learning_experiment_best_hyperparameters.py
|   |-- files
|   |   |-- baseline_experiment
|   |   |   |-- hyperparameters_all_features
|   |   |   |-- hyperparameters_selected_features
|   |   |   |-- performances
|   |   |-- baseline_experiment_best_hyperparameters
|   |   |   |-- hyperparameters_all_features
|   |   |   |-- hyperparameters_selected_features
|   |   |   |-- performances
|   |   |-- features
|   |   |   |-- dataset
|   |   |   |-- dataset_Task
|   |   |-- tasks_experiment
|   |   |   |-- hyperparameters_all_features
|   |   |   |-- hyperparameters_selected_features
|   |   |   |-- performances
|   |   |-- tasks_experiment_best_hyperparameters
|   |   |   |-- hyperparameters_all_features
|   |   |   |-- hyperparameters_selected_features
|   |   |   |-- performances
|   |   |-- tasks_final_ensemble_learning_experiment
|   |   |-- tasks_final_ensemble_learning_experiment_best_hyperparameters
|   |-- utils
|   |   |-- skfeature
|   |   |-- config.yml
|   |   |-- constants.py
|   |   |-- feature_selection.py
|   |   |-- models_and_hyperparameters.py
|   |   |-- utilities.py
|   |-- main_demonstrator.py
|   |-- main_experiments.py
```

The developed code is divided in the following folders/files:

- The “experiments” folder contains all the files relative to the experiments. In particular, it contains:
    - baseline_experiment.py performs Baseline experiments
    - baseline_experiment_best_hyperparameters.py performs Baseline experiments with the best hyperparameters
    - tasks_experiment.py performs Tasks experiments
    - tasks_experiment_best_hyperparameters.py performs Tasks experiments with the best hyperparameters
    - tasks_final_ensemble_learning_experiment.py performs Ensemble Learning experiments
    - tasks_final_ensemble_learning_experiment_best_hyperparameters.py performs Ensemble Learning experiments with the
      best hyperparameters
- The “files” folder contains dataset PaHaW, models' hyperparameters and experiments' performances. In particular, it
  contains:
    - baseline_experiment folder contains hyperparameters used for the training of Baseline experiment and it lies on
      all features and selected features. Indeed, it contains also relative performances obtained by the models'
      configurations
    - baseline_experiment_best_hyperparameters folder contains the best hyperparameters used for the training of
      Baseline experiment and it lies on all features and selected features. Indeed, it contains also relative
      performances obtained by the models' configurations
    - features folder contains the dataset PaHaW with all the features at once and with all tasks it is composite of
    - tasks_experiment folder contains hyperparameters used for the training of Tasks experiment and it lies on all
      features and selected features. Indeed, it contains also relative performances obtained by the models'
      configurations
    - tasks_experiment_best_hyperparameters folder contains the best hyperparameters used for the training of Tasks
      experiment and it lies on all features and selected features. Indeed, it contains also relative performances
      obtained by the models' configurations
    - tasks_final_ensemble_learning_experiment folder contains performances obtained by the experts' combination of
      Ensemble Learning experiment
    - tasks_final_ensemble_learning_experiment_best_hyperparameters folder contains the best performances obtained by
      the experts' combination of Ensemble Learning experiment
- The “utils” folder contains all utility useful for others files.py. In particular, it contains:
    - skfeature folder contains other function useful to implement Filter techniques
    - config.yml provides paths
    - constants.py provides useful constants to use in different files.py
    - feature_selection.py applies Feature Selection techniques
    - models_and_hyperparameters.py fetches configuration Model/Feature-Selection-Techniques with hyperparameters
    - utilities.py provides utilities used from different files.py
- The file main_demonstrator.py which fetches dataset and starts demonstrator.
- The file main_experiments.py which fetches dataset and starts experiments.

# How to run

1. Run the main_demonstrator.py or main_experiments.py
2. In case of running main_demonstrator.py it automatically shows the results, otherwise main_experiments.py requires to
   respond different questions in order to perform desired experiments
