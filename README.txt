
A Language Detection Neural Network with Vector Hashing.

Artificial Intelligence Project.
 
I was required to use the Encog 3.x API to develop a multilayer neural network capable of
detecting the language of text in a file. The text should be parsed, converted into n-grams and
then hashed to a vector before being fed as input to a neural network.

 Project Overview.

This application covers the minimum requirements outlined in the project specification.
As a user is being presented with a simple menu at the start of application ,he/she selects
options:

1) to prepare data required to build a model for training, concretely parsing sample texts into n-grams and hashing, converting shingles
into digital text represention and putting into a vector.A user is given an opportunity to select a vector and n-grams size by doing that training time 
and accuracy of prediction may vary so for the best results the recommended numbers should be used. For the vector size any unsigned integers are acceptable, for 
n-grams a user can use any single unsigned integers or an array of unsigned integers separated by comma. After reading in a n-grams size is being written to file,
because it is stored in an array of integers, it gets serialized as an object. This useful feature prevents a user from entering a wrong n-gram size in step 3 
for text parsing from file.
 
2) to build a new neural network. At this point a neural network has all required data to process. In case a user would skip step 1 to prepare data, 
an error is being thrown and handled promting user to go back to prepare data.

3) to predict a language. A user is asked to enter a path to a text file with sample text to predict a language.If the file not found a user being promted to 
enter a valid path. The application proceeds trying to load a saved trained neural network model if success n-gram size is read back in from a saved file,
because vector size is mapped 1:1 to input nodes, we can use method getInputCount() of BasicNetwork class to retrieve that.

 Design rationale.

• The n-gram or other hashing approach used by your application.

  "The total number of possible n-grams in an n-spectrum is ∑n, where ∑ is the number of
symbols in the alphabet for the text, e.g. the full n-spectrum for 5-gram lower-case words in
English has 265 = 11,881,376 elements. In practical terms, an n-spectrum will not be fully
realized, as the vast majority of n-gram combinations will not be present in text. It should also
be clear that the frequency of a 2-gram appearing in a text will be much greater than that of a
5-gram. For example, if we only include lower-case characters in English, each character will
appear with a probability of 1/26 = 0.038. A 2-gram will therefore occur with a probability of
0.038 * 0.038 = 0.001444 and a 5-gram with a probability of 0.038 * 0.038 * 0.038 * 0.038 *
0.038 = 0.0385 = 7.92x10-8." 
			    Prof. John Healy.

I found that combinations of 1 and 2 n-grams put together into a vector of fixed size work best. Before proccesing i had removed all punctuations and numeric 
symbols, replaced white space with underscore "_". Because we have about 50 samples per a language and by combining the highest frequency of 1 and 2-gram 
appearing in a text after processing we would double those samples, the vector fixed size stores now 100 samples the most frequently used n-grams.Not exactly
n-grams text but rather the frequency of appearance those n-grams as an integer number. For example by using only 3-gram in 3 min. of training the accuracy on 
test data got 75 % ,although solely 2-gram got 85 % on the other hand a combination of 1,2-gram got 90 %.


• The size of the hashing feature vector.

The size of the hashing vector eventually indicates how many input neurons being used. It is equal to the number of features (columns) in the
data which maps 1:1. By significantly increasing vector size (features) we create a lot of neurons empty without any data but those neurons must 
be processed in the same way consuming time, the data is spead over but having small vector size makes data being squashed.For example having 1000 neurons 
in 3 min. made only 2 epoches with accuracy on test data got 65 % and having only 50 neurons (vector size) made 62 epoches in 3 min. with accuracy 72 %. And in my case
i used input neurons to calculate hidden nodes (modified geometric pyramid rule) hiddenNodes = (int) (Math.sqrt(input * output)) / 2 , so having big vector
size would increase the number of hidden nodes in my hidden layer.My best vector size is 350.

• The number of hidden layers and the number of nodes in each layer.

 For my neural network having one hidden layer is sufficient, firstly we don't have a lot of data but adding more hidden layers will increase the amount of time to train
and test as each new layer increase the computational complexity expotentially.Secondly it may or may not improve the accuracy of the neural network.Being limited in 3 min.
to reach  98 % accuracy it is not feasible. For example with a set up as 350 vector size, 1,2-grams with one hidden layer got accuracy 90 % in 3 min. but adding second
hidden layer got only 79 %.

  The number of neurons in input layer is equal to the number of vector features (columns) in the
   	data 1:1 mapping. Some neural network configurations add one additional node for a bias term.

  The number of neurons in output layer is determined by the chosen model configuration.
	• Classification: one node / class, e.g. 235 nodes for 235 languages.
	• Prediction (Regression): a single node, e.g. stock price.

  Computing the correct number of nodes in a hidden layer is a black art.
	Too many nodes is called overfitting and results in a network that is difficult to train without a very large data set. Nodes are starved.
	Underfitting results in too few neurons in the hidden layers to adequately detect the signals in a complicated data set. Nodes get saturated.
	For my application i found that (geometric pyramid rule) hiddenNodes = (int) (Math.sqrt(input * output)) works best, i divided by 2 and accuracy 
	jumped from 72 % to 90 %, divided by 3 and improved to 91.5 % 



• The overall neural network topology.
	
	-Input layer has no activation function, bias(true), input nodes equals vector size.
	 Input layer rarely includes computing neurons.Does not process input patterns.
	 Some neural network configurations add one additional node for a bias term.
	-Hidden layer has ActivationTANH, bias(true), hidden nodes are calculated with (geometric pyramid rule) (int)(Math.sqrt(input * output))/2;
	-Output layer has ActivationSoftMax, bias(false), output nodes (235) the number of languages (classes);
 	

• The activation functions used in each layer.
	
	-Input layer has no activation function, the first layer is fed the raw input and weights which are then transformed via the activation function at the subsequent neurons.
	-Hidden layer has hyperbolic tangent activation function,among relu and sigmoid, tanh function performs better in a time limit of 3 min. The hyperbolic tangent function 
	 	is the default activation function for Encog.
	-Output layer has softMax activation function, it always returns a probability distribution over the target classes in a multiclass classification problem.

• Conclusion.
	
	This project gave me a lot of ground to experiment with different settings. Once after running down a training error stopped decreasing and started going up again,
	i realized that may be the case with a learning rate being too big and convergence to global minimum is overshot, i started tweaking ResilientPropagation parameters
	default updateRate was 0.1, i changed to 0.01 and the largest amount that the update values can step was 50, mine is 0.1. And that actually improved the whole 
	network as a training error steadily went down and accuracy is achieved ~90 % in 3 min.

