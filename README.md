# Тестовое задание на позицию Computer Vision Data Scientist
## Решение Трифонов Р.С.

### Задачи 

> **Необходимо построить классификатор деталей по фотографиям, который будет предсказывать один из двух классов: «маленькая» или «большая» деталь на изображении.**

- Необходимо добиться максимальной точности предсказания классов.

### Стратегия решения 

В качестве исходных данных предоставлен маленький набор данных, содержаний фотографии «больших» и «маленьких» деталей.  Учитывая ограниченность тренировочных данных в решении, используются три основных стратегии для поиска лучшего решения. 

1. Обучение простой модели Tiny VGG  и ее последующее обучение на имеющихся данных. 
2.  Дообучение наиболее популярных моделей с помощью подходов fine - tuning model и feature extraction
3.  В качестве популярных моделей классификации для дообучения выбраны: 
  - VGG-16
  - ResNet50
  - Inception v3
  - Efficientnet


### Промежуточные результаты

- [Tiny VGG](https://github.com/rttrif/Test_task_computer_vision/blob/main/Part_classifier_Tiny_VGG.ipynb)
- [Fine - tuning models](https://github.com/rttrif/Test_task_computer_vision/blob/main/Part_classifier_FT.ipynb)
- [Feature extraction](https://github.com/rttrif/Test_task_computer_vision/blob/main/Part_classifier_FE.ipynb)
