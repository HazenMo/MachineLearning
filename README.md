# Overview

I am using Machine Learning to interpret a data set of hand written digits (numbers between 0 and 9). The data set is provided by MNST but at the end of the program there is an opportunity to introduce your own "hand written" digit for the model to interpret. 

I created this program not so much to interpret a data set, but mostly to learn about neural networks and machine vision. I wanted to explore the creation and training of a nueral network, and then to implement that training to do some deductions. 


[Software Demo Video](http://youtube.link.goes.here)

# Data Analysis Results

The neural network usually finishes it's training with an accuracy of about 97% to 98%. Though with experimentation usually it does not do as well when the completely new instance is interpreted. I believe this is a by product of overfitting. 

# Development Environment

I used VS Code as the IDE for this project.

I wrote this project in Python and used a number of libraries to make this project work. Fist and foremost I used tensorflow to build my neural network. I also used numpy and matplotlib to create the displays that show the data set and labels so you get an idea what is being fed into the network. Finally I used pygame and PIL so the user can create their own piece of data to test the network. Pygame creates the drawn on image, and PIL allows the program to format the image into something the network can actually interpret.

# Useful Websites

{Make a list of websites that you found helpful in this project}
* [TensorFlow](https://www.tensorflow.org/tutorials)
* [GeeksforGeeks](https://www.geeksforgeeks.org/)

# Future Work

* I definitly want to fix the overfitting problem that the program has. I have a few ideas of how I might do this but the top of my list is currently to use data generation tools to modify the data set, so that it might get used to numbers that are shown a little differently.
* I also want to figure a way to save the trained network so that it has the ability to be used without training it, every singe time.
* Finally I think it would be awesome to expand the network beyond just digits to handwritten letters as well, this would be as simple as adding more data to the training, but I would eventually want to put it to work interpreting handwritten words and phrases from say a phone camera.