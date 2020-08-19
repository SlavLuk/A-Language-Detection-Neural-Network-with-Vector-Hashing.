package ie.gmit.sw;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.PrintWriter;
import java.nio.charset.StandardCharsets;
import java.text.DecimalFormat;

import org.encog.ml.data.MLData;
import org.encog.ml.data.basic.BasicMLData;

public class TextParser {

	private DecimalFormat decimalFormat;
	private double[] langEncoded;
	private int[] kmer;
	private double[] vector;
	private PrintWriter out;
	private File fname;

	public TextParser(int input, int[] kmer, int langSize, File fname) throws IOException {

		this(input, kmer, fname);

		this.langEncoded = new double[langSize];
		this.decimalFormat = new DecimalFormat("###.###");
		this.out = new PrintWriter(new BufferedWriter(new FileWriter("./data.csv")));
	}

	public TextParser(int size, int[] kmer, File fname) {

		this.vector = new double[size];
		this.kmer = kmer;
		this.fname = fname;
	}

	public void readFile() {

		try (BufferedReader br = new BufferedReader(
				new InputStreamReader(new FileInputStream(fname), StandardCharsets.UTF_8));

		) {

			String line = null;

			// read file by lines
			while ((line = br.readLine()) != null) {

				// split line into two dataset
				String[] record = line.trim().split("@");

				// if line has no @ sign keep going
				if (record.length != 2) {

					continue;
				}

				parse(record[0], record[1]);

			}

			out.close();

		} catch (Exception e) {
			e.printStackTrace();
		}

	}

	public MLData parse() {

		init(vector);

		try (BufferedReader br = new BufferedReader(
				new InputStreamReader(new FileInputStream(fname), StandardCharsets.UTF_8));

		) {
			String line = null;

			// read file by lines
			while ((line = br.readLine()) != null) {

				nGrams(getText(line), kmer);

			}

			// normalize vetor data before returning
			vector = Utilities.normalize(vector, 0, 1);

		} catch (Exception e) {

			System.err.println("Error occured " + e.getMessage());

		}

		return new BasicMLData(vector);
	}

	// overloading method
	public void parse(String text, String lang) {

		Language language = Language.valueOf(lang);

		// initialize to 0
		init(vector);
		init(langEncoded);

		// get ordinal and set to 1 for a particular language
		langEncoded[language.ordinal()] = 1;

		nGrams(getText(text), kmer);

		// normalize vetor data before returning
		vector = Utilities.normalize(vector, 0, 1);

		//write to data.csv file
		for (int i = 0; i < vector.length; i++) {

			out.write(decimalFormat.format(vector[i]) + ",");

		}

		for (int i = 0; i < langEncoded.length; i++) {

			out.write(langEncoded[i] + "");

			if (i == langEncoded.length - 1) {

				out.write("\n");

				break;

			}

			out.write(",");

		}

	}

	// parse text into n size kmer and store hashed code into feature vector
	private void nGrams(String t, int[] kmer) {

		for (int n : kmer) {

			for (int i = 0; i <= t.length() - n; i++) {

				String s = t.substring(i, i + n);

				// convert string into hashcode int
				int index = s.hashCode() % vector.length;

				vector[index]++;

			}

		}
	}

	private String getText(String text) {

		// regexp used to remove numeric,punctuations items
		 String t = text.replaceAll("\\p{P}", "")
						.toLowerCase()
						.replaceAll("\\d", "")
						.replaceAll("  ", " ")
						.replaceAll(" ", "_");

		return t;
	}

	// initialize vector feature array to 0
	private void init(double[] arr) {

		for (int i = 0; i < arr.length; i++) {

			arr[i] = 0;
		}

	}

}