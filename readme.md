"Soji-DNN Is Going to Ingenium 2023 :)"
-GG 

# YOLOv7 Garbage Detection Inference and Model by GG
##### This Git repository is the main repo for Project Soji DNN/Ingenium 2023.

In here, we make a model based on WongKinYiu's YOLOv7 ML Model, which makes a Bounding Box Prediction by only looking at the image once, where the name "You Only Look Once" came from.

The purpose of this project is to Create an adequate model that can detect pieces of garbage based on an image, which will later be adapted into the project. The enviroment for the project will be a blank, preferrably metal/aluminum table, which will allow a clear background for the arm to detect trash.
Currently, we have:

--Inference Code

--Basic Models for Trash Detection (see prebuilts/custom)

And, we currently need both better hardware and a way to convert the model to OpenVINO for Intel NCS2 Inference (Both coming soon!)

Right now, all custom models are trained with our Aquatrash dataset, but when we get the CUDA Hardware, we will start production on a model based on the Trash-Filter Model (6000+ images!!!)

# TODO:

Get v7t_p5hyps running on ncs2

Make v7t model with tiny hyps

others idk :)))))
