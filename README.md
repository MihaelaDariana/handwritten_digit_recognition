# handwritten_digit_recognition

This project recognizes digits written by hand. It's implemented from this tutorial: https://www.youtube.com/watch?v=bte8Er0QhDg
For this project, the following modules are used: tensorflow, keras, numpy, opencv-python, matplotlib. This can be installed using pip install command.

Here are 3 parts. 
1. Load and train data. We have to run the load_and_train.py script.
   In this script, we get the train and test data directly from tensorflow. We reshape de data and then we create the model and add layers to it, and then the model is trained.
   

![ep7](https://github.com/MihaelaDariana/handwritten_digit_recognition/assets/80625876/6d5a9695-0bbf-4b09-a967-7471c1b03fcd)

2. Test the model. We have to run the test.py script
   Here we run the model with the test data and verify the accuracy and loss.
   ![loss_accuracy](https://github.com/MihaelaDariana/handwritten_digit_recognition/assets/80625876/c84d2d9b-b9e2-44f4-8137-85e41e1bdf9c)

3. Here we test the model on custom handwritten digits. We run test_on_my_digits.py.
   In the "digits" directory are 12 photos 28*28 pixels with digits on them. We apply the model to each photo to see how is the accuracy.
   
![right digit](https://github.com/MihaelaDariana/handwritten_digit_recognition/assets/80625876/e834d494-bdfc-44ef-a0c8-8f9c28e5e54c)
![wrong_digit](https://github.com/MihaelaDariana/handwritten_digit_recognition/assets/80625876/abb44e29-5d2d-465f-a4ce-27515c0edca1)
