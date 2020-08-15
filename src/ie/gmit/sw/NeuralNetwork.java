package ie.gmit.sw;

import java.io.File;
import org.encog.engine.network.activation.ActivationSoftMax;
import org.encog.engine.network.activation.ActivationTANH;
import org.encog.ml.data.MLData;
import org.encog.ml.data.MLDataPair;
import org.encog.ml.data.MLDataSet;
import org.encog.ml.data.buffer.MemoryDataLoader;
import org.encog.ml.data.buffer.codec.CSVDataCODEC;
import org.encog.ml.data.buffer.codec.DataSetCODEC;
import org.encog.ml.data.folded.FoldedDataSet;
import org.encog.ml.train.MLTrain;
import org.encog.neural.networks.BasicNetwork;
import org.encog.neural.networks.layers.BasicLayer;
import org.encog.neural.networks.training.cross.CrossValidationKFold;
import org.encog.neural.networks.training.propagation.resilient.ResilientPropagation;
import org.encog.util.csv.CSVFormat;

public class NeuralNetwork {

	private int input; // Change this to the number of input neurons
	private int output;// Change this to the number of output neurons
	private String csv;
	private int hiddenNodes;
	private static final String SAVED_NN = "./test.nn";

	public NeuralNetwork(int in, int output, String csvFile) {

		this.input = in;
		this.output = output;
		this.csv = csvFile;

	}

	// build neural network
	public void networkBuilder() {

		BasicNetwork basicNN = getNetwork();
		MLDataSet dataSet = loadData();
		CrossValidationKFold crossFold = crossValidation(dataSet, basicNN);
		trainNN(crossFold, basicNN);
		testNN(basicNN, dataSet);

	}

	private BasicNetwork getNetwork() {

		// calculate nodes for hidden layer with modified geometric pyramid rule
		 hiddenNodes = (int) (Math.sqrt(input * output)) / 2;

		// Configure the neural network topology.
		BasicNetwork network = new BasicNetwork();
		network.addLayer(new BasicLayer(null, true, input));
		network.addLayer(new BasicLayer(new ActivationTANH(), true, hiddenNodes));
		network.addLayer(new BasicLayer(new ActivationSoftMax(), false, output));
		network.getStructure().finalizeStructure();
		network.reset();

		return network;
	}

	private MLDataSet loadData() {

		// Read the CSV file "data.csv" into memory.
		DataSetCODEC dsc = new CSVDataCODEC(new File(csv), CSVFormat.DECIMAL_POINT, false, input, output, false);
		MemoryDataLoader mdl = new MemoryDataLoader(dsc);
		MLDataSet trainingSet = mdl.external2Memory();

		return trainingSet;
	}

	private CrossValidationKFold crossValidation(MLDataSet dataSet, BasicNetwork basicNN) {

		// train using K-Fold cross validation
		FoldedDataSet folded = new FoldedDataSet(dataSet);
		MLTrain train = new ResilientPropagation(basicNN, folded, 0.01, 0.1);
		CrossValidationKFold cv = new CrossValidationKFold(train, 5);

		return cv;
	}

	private void trainNN(CrossValidationKFold crossFold, BasicNetwork basicNN) {

		// Train the neural network

		int epoch = 0;
		
		System.out.println("[INFO] Training... ");
		
		long startTime = System.currentTimeMillis();
		
		long finishTime;
		
		do {
			
			crossFold.iteration();
			
			epoch++;
			
			System.out.println("Epoch #" + epoch + " Error:" + crossFold.getError());

			finishTime = System.currentTimeMillis();
			
		} while ((finishTime - startTime) / 1000 < 175);

		long totalTime = finishTime - startTime;

		Utilities.saveNeuralNetwork(basicNN, SAVED_NN);

		crossFold.finishTraining();

		System.out.println("\n[INFO] Training has completed in  " + (totalTime / 1000) / 60 + " min. "
				+ (totalTime / 1000) % 60 + " seconds. In " + epoch + " epochs with error = " + crossFold.getError());

		System.out.println("[INFO] Neural Network has 3 layers.");
		System.out.println("[INFO] Input layer has no activation function, bias (true), input nodes "+input);
		System.out.println("[INFO] Hidden layer has TANH activation function, bias (true),hidden nodes " + hiddenNodes);
		System.out.println("[INFO] Output layer has SoftMax activation function,bias (false),output nodes " + output);
		

	}

	private void testNN(BasicNetwork basicNN, MLDataSet dataSet) {

		double correct = 0;
		double total = 0;

		//traverse over dataset
		for (MLDataPair pair : dataSet) {

			int resultIndex = -1;
			int idealIndex = -1;
			double max = 0;

			total++;
            
			MLData outputData = basicNN.compute(pair.getInput());

			double[] out = outputData.getData();
			double[] ideal = pair.getIdeal().getData();

			//find the closest to 1 and retrieve that index
			for (int i = 0; i < out.length; i++) {

				if (out[i] > max) {

					max = out[i];

					resultIndex = i;
				}

			}

			//get index an array of 1
			for (int i = 0; i < ideal.length; i++) {

				if (ideal[i] == 1) {

					idealIndex = i;

				}
			}
            
			//compare indexes
			if (idealIndex == resultIndex) {
				correct++;
			}
		}

		System.out.println("[INFO] Testing has completed. Accuracy = " + ((correct / total) * 100) + " Total : " + total
				+ " Correct : " + correct);

	}

}
