package model;

import java.util.ArrayList;

import utility.FileOneByOneLineWriter;
import utility.FileReaderAndWriter;
import utility.ItemWithValue;
import nlp.Corpus;
import nlp.Vocabulary;

/**
 * This class prints the outputs of a topic model.
 * 
 * For each model, we print:
 * 
 * 1. Model parameters
 * 
 * 2. z[][]
 * 
 * 3. Topic-word distribution
 * 
 * 4. Top words under each topic.
 * 
 * 5. Vocabulary
 */
public class ModelPrinter {
	// Suffix for model parameters.
	public static final String modelParamSuffix = ".param";
	// Suffix for topic word assignment file.
	public static final String tassignSuffix = ".tassign";
	// Suffix for document topic distribution.
	public static final String documentTopicDistSuff = ".dtopicdist";
	// Suffix for topic word distribution.
	public static final String topicWordDistSuff = ".twdist";
	// Suffix for file containing top words per topic.
	public static String twordsSuffix = ".twords";
	// Suffix for file containing the documents in the corpus.
	public static String docsSuffix = ".docs";
	// Suffix for file containing the vocabulary.
	public static String vocabSuffix = ".vocab";
	// Suffix for file containing the knowledge.
	public static String knowledgeSuffix = ".knowl";

	private TopicModel model = null;

	public ModelPrinter(TopicModel model2) {
		model = model2;
	}

	/**
	 * Print the model.
	 */
	public void printModel(String outputDirectory) {
		try {
			String domain = model.param.domain;
			printModelParameters(model.param, outputDirectory + domain
					+ modelParamSuffix);
			printTopicWordAssignment(model.getTopicWordDistributionOne(), model.corpus, outputDirectory
					+ domain + tassignSuffix);
			printDocumentTopicDistribution(model.getDocumentTopicDistrbution(), outputDirectory + domain + documentTopicDistSuff);
			printTopicWordDistribution(model.getTopicWordDistribution(),
					outputDirectory + domain + topicWordDistSuff);
			ArrayList<ArrayList<ItemWithValue>> topWordsUnderTopics = model
					.getTopWordStrsWithProbabilitiesUnderTopics(model.param.twords);
			printTopWordsUnderTopics(topWordsUnderTopics, outputDirectory
					+ domain + twordsSuffix);
			printDocs(model.corpus.docs, outputDirectory + domain + docsSuffix);
			printVocabulary(model.corpus.vocab, outputDirectory + domain
					+ vocabSuffix);

			// For knowledge-based topic models only.
			model.printKnowledge(outputDirectory + domain + knowledgeSuffix);
		} catch (Exception ex) {
			System.out.println("Error while printing the topic model: "
					+ ex.getMessage());
			ex.printStackTrace();
		}
	}

	private void printModelParameters(ModelParameters param, String filePath) {
		param.printToFile(filePath);
	}

	private void printTopicWordAssignment(double[][][] phi, Corpus corpus,
			String filePath) {
		assert (phi != null && phi.length != 0 && phi[0].length != 0 && phi[0][0].length != 0) : "The array z is not correct!";

		FileOneByOneLineWriter writer = new FileOneByOneLineWriter(filePath);

		int D = phi.length;
		for (int d = 0; d < D; ++d) {
			StringBuilder sbLine = new StringBuilder();
			int N = phi[d].length;
			for (int n = 0; n < N; ++n) {
				for (int k = 0; k < phi[d][n].length; k++) {
					sbLine.append( phi[d][n][k] + " ");
					writer.write(sbLine.toString());
				}
				writer.write("$$$$$");
			}
			writer.writeLine();
		}
		writer.close();
	}

	private void printDocumentTopicDistribution(double[][] dist, String filePath) {
		assert (dist != null && dist.length != 0 && dist[0].length != 0) : "The document topic distribution is not correct!";

		FileOneByOneLineWriter writer = new FileOneByOneLineWriter(filePath);

		int D = dist.length;
		for (int d = 0; d < D; ++d) {
			StringBuilder sbLine = new StringBuilder();
			int T = dist[d].length;
			for (int t = 0; t < T; ++t) {
				sbLine.append(dist[d][t] + " ");
			}
			writer.writeLine(sbLine.toString().trim());
		}
		writer.close();
	}
	
	private void printTopicWordDistribution(double[][] dist, String filePath) {
		assert (dist != null && dist.length != 0 && dist[0].length != 0) : "The topic word distribution is not correct!";

		FileOneByOneLineWriter writer = new FileOneByOneLineWriter(filePath);

		int T = dist.length;
		for (int t = 0; t < T; ++t) {
			StringBuilder sbLine = new StringBuilder();
			int V = dist[t].length;
			for (int w = 0; w < V; ++w) {
				sbLine.append(dist[t][w] + " ");
			}
			writer.writeLine(sbLine.toString().trim());
		}
		writer.close();
	}

	private void printTopWordsUnderTopics(
			ArrayList<ArrayList<ItemWithValue>> topWordsUnderTopics,
			String filepath) {
		StringBuilder sbOutput = new StringBuilder();

		// Print out the first row with "Topic k".
		for (int k = 0; k < topWordsUnderTopics.size(); ++k) {
			sbOutput.append("Topic " + k +": ");
			for (ItemWithValue array : topWordsUnderTopics.get(k)) {
				sbOutput.append(array.getIterm()+" ");
			}
			sbOutput.append(System.getProperty("line.separator"));
			sbOutput.append(System.getProperty("line.separator"));
		}
		
		FileReaderAndWriter.writeFile(filepath, sbOutput.toString());
	}

	private void printDocs(int[][] docs, String filePath) {
		StringBuilder sbOutput = new StringBuilder();
		for (int[] doc : docs) {
			StringBuilder sbLine = new StringBuilder();
			for (int word : doc) {
				sbLine.append(word + " ");
			}
			sbOutput.append(sbLine.toString().trim());
			sbOutput.append(System.getProperty("line.separator"));
		}
		FileReaderAndWriter.writeFile(filePath, sbOutput.toString());
	}

	private void printVocabulary(Vocabulary vocab, String filepath) {
		vocab.printToFile(filepath);
	}
}
