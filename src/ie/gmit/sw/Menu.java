package ie.gmit.sw;

import java.io.*;
import java.util.Scanner;
import org.encog.Encog;
import org.encog.ml.data.MLData;
import org.encog.neural.networks.BasicNetwork;

public class Menu {

	private Scanner scan = new Scanner(System.in);
	private int choice;
	private int[] kmer;
	private int vectorSize;
	private BasicNetwork savedNetwork;
	private MLData dataVector;
	private Language[] langs;
	private boolean keepRunnig = true;
	private File fname = new File("./wili-2018-Small-11750-Edited.txt");
	private final static int OUTPUT = 235;
	private final static String CSV_FILE = "./data.csv";

	public void show() throws Exception {

		System.out.println("\n***********************************************************");
		System.out.println("* A Language Detection Neural Network with Vector Hashing.*");
		System.out.println("***********************************************************");
		System.out.println("Please enter a number of your choice.");

		do {

			System.out.println("\nPlease enter 1) to prepare data for neural network.");
			System.out.println("Please enter 2) to build a new neural network.");
			System.out.println("Please enter 3) to predict a language.");
			System.out.println("Please enter 4) to quit.");

			String str = scan.next();
			
			try {
				
				choice = Integer.parseInt(str);
				
			} catch (Exception e) {
				
				System.err.println("Invalid input...");
			}

			switch (choice) {

			case 1:

				while (!getVectorSize());
				
				while (!getKmerSize());
				
				System.err.println("Processing \"" + fname.getPath() + "\" file...");
				
				//put main thread to sleep for 5 sec. to get data processed
				Thread.sleep(5000);

				dataParse();
				
				System.err.println("Done processing...");

				break;

			case 2:

				try {

					new NeuralNetwork(vectorSize, OUTPUT, CSV_FILE).networkBuilder();

				} catch (Exception e) {

					System.err.println("Sorry, you don't have prepared data to process...\nPlease go back to step 1.");

				}

				Encog.getInstance().shutdown();

				break;

			case 3:
				
				langs = Language.values();
				
				boolean loop;
				
				File f;
				
				do {
					loop = false;

					System.out.println("Please enter a file path to parse.");

					String fileName = scan.next();

					 f = new File(fileName);

					if (!f.exists()) {

						System.err.println("Sorry, no such file exists...");

						loop = true;
					}

				} while (loop);
				

				predict(f);

				break;
				
			case 4:

				keepRunnig = false;
				
				break;

			default:

				System.err.println("No such choice...");
				
				break;
			}

		} while (keepRunnig);

	}

	private void predict(File f) {
		
		try {
			
			savedNetwork = Utilities.loadNeuralNetwork("./test.nn");
			
		} catch (Exception e) {
			
			System.err.println("No saved network, Please go back to step 1.");
			
			return;
		}
		
		int[]k = readKmer("./kmer.txt");
		
		dataVector = new TextParser(savedNetwork.getInputCount(),k , f).parse();

		int resultIndex = -1;
		double max = 0;

		MLData output = savedNetwork.compute(dataVector);

		double[] out = output.getData();

		for (int i = 0; i < out.length; i++) {
			
			if (out[i] > max) {

				max = out[i];
				
				resultIndex = i;
			}
		}
		
		if(k!=null) {
			
			System.err.println("Predicted Language is : " + langs[resultIndex]);
		}

	

	}

	private void dataParse() {

		
		
		new Thread(new Runnable() {

			@Override
			public void run() {

				try {

		        	new TextParser(vectorSize, kmer, OUTPUT, fname).readFile();
				
			
				} catch (IOException e) {

					System.err.println("Error occured.");
				}
			}
		}).start();
		
	}

	private int[] getKmer(String str) {

		String[] kmerStr = str.split(",");

		int[] k = new int[kmerStr.length];

		int i = 0;

		for (String s : kmerStr) {

			k[i] = Integer.parseInt(s);

			i++;
		}
		
		saveKmer(k);
		
		
		return k;
	}
	
	private int[] readKmer(String fname) {
		
		int[] size = null;
		
		try(
				ObjectInputStream oi = new ObjectInputStream( new FileInputStream(new File(fname)));){

			// Read objects
			size = ( int[]) oi.readObject();
			
			
		}catch(Exception e) {
			
		}
		return size;
	}
		
	
	//serialize an array of kmer
	private void saveKmer(int[]size) {
	
		try(
				ObjectOutputStream o = new ObjectOutputStream(new FileOutputStream(new File("./kmer.txt")));


			){	o.writeObject(size);
			
		}catch(Exception e) {
			
		}
		
	}

	private boolean getKmerSize() {

		System.out.println("Please enter k-mer size one digit (e.g. 3) or an array "
				+ "separated by comma (e.g. 1,2,3) (Recommended 1,2)");

		try {
			String kmerList = scan.next();

			kmer = getKmer(kmerList);

		} catch (Exception e) {

			System.err.println("Invalid input");

			return false;
		}

		return true;

	}

	private boolean getVectorSize() {

		try {

			System.out.println("Please enter vector size (Recommended 350)");

			String size = scan.next();

			vectorSize = Integer.parseInt(size);

		} catch (Exception e) {

			System.err.println("Invalid input");

			return false;
		}

		return true;

	}
}
