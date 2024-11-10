# crop-scan
An intelligent system for early detection of plant diseases using image recognition technology.


# Inspiration
The idea for this project was inspired by the global challenge of crop diseases and how they affect food production and sustainability. We noticed that many farmers and gardeners struggle with early detection of diseases in their plants, often relying on guesswork or costly professional help. With the rise of AI and image recognition technology, we saw an opportunity to build a tool that could simplify this process, making plant disease detection more accessible to everyone.

# What We Learned
Throughout the project, we gained a deep understanding of both the complexities of plant diseases and the technicalities of building an image recognition model. We learned how different diseases manifest visually on plant leaves and how to train a machine learning model to detect these patterns. We also explored various machine learning techniques, including image preprocessing, data augmentation, and model optimization, to improve accuracy and speed.

Additionally, we discovered the importance of clean, high-quality data in building a robust model. Sourcing and organizing data from multiple datasets posed some unique challenges, but it also taught us the value of persistence and creativity in problem-solving.

# How We Built It
We started by gathering a dataset of plant leaf images, with each image labeled as either healthy or diseased. We used popular machine learning libraries such as TensorFlow and Keras to build our image classification model. Here's a breakdown of the steps we followed:

- Data Collection: We sourced leaf images from various public datasets, ensuring a wide variety of plants and diseases were covered.
-Data Preprocessing: Images were resized, augmented, and normalized to standardize the inputs for the model.
- Model Training: We utilized Convolutional Neural Networks (CNNs), a popular deep learning architecture for image recognition tasks, to train the model.
- Testing & Optimization: We fine-tuned the model, adjusting hyperparameters and experimenting with different architectures to improve its performance.
- User Interface: We developed a simple, user-friendly interface where users can upload an image of a plant leaf and instantly receive a diagnosis.
