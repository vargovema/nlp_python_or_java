# Classifying Stack Overflow posts using Machine Learning and Natural Language Processing

Stack Overflow is a platform for people to ask and answer questions about computer programming. The site uses a system of tags to label posts. These tags help users search for questions. Often users do not tag their posts but NLP can help. 

In this task, your goal will be to train a machine learning classifier to distinguish between questions tagged *python* and *java* using the text of the question. The resulting classifier represents a first step towards a system that can automatically categorize new incoming posts.

I have provided you with a random sample of 1000 posts tagged python and 1000 posts tagged java. Note that none of the posts in your sample have been tagged with both python and java. To make the task more challenging, I have removed the substrings python, java, and py from the questions.