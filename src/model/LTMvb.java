package model;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Map;
import java.util.PriorityQueue;
import java.util.Random;

import knowledge.KnowledgeAsMustClustersGenerator;
import knowledge.MustLink;
import knowledge.MustLinks;
import nlp.Corpus;
import nlp.Vocabulary;
import utility.FileReaderAndWriter;

/**
 * This implements the LDA model (Blei et al., 2003).
 */
public class LTMvb extends TopicModel {

	double[][] lambda;// Variational Parameters for topic-word distribution K*V
	double[][] gamma;// Variational Parameters for doc-topic distribution M*K
	double[][][] phi;// Variational Parameters for doc-word-topic distribution

	
	/******************* Knowledge *********************/
	// The must-links for each topic.
	HashMap<Integer, MustLinks> topicToMustlinksMap = null;
	// The topic model lists that the knowledge is extracted from.
	ArrayList<TopicModel> topicModelListForKnowledgeExtraction = null;
	// urn_Topic_W1_W2_Value.get(t).get(w1).get(w2): the urn values of pair of
	// words (w1, w2) under a topic t.
	// Note that the urn matrix does not contain values for same pair of words
	// (w, w).
	private HashMap<Integer, HashMap<Integer, HashMap<Integer, Double>>> urn_Topic_W1_W2_Value = null;

	
	
	/**
	 * Create a new topic model with all variables initialized. The z[][] is
	 * randomly assigned.
	 */
	public LTMvb(Corpus corpus2, ModelParameters param2) {
		super(corpus2, param2);

		initializeModel(corpus2, param2);
		
		TopicModel topicmodel_currentDomain = findCurrentDomainTopicModel(param.topicModelList_LastIteration);
		initializeFirstMarkovChainUsingExistingZ(topicmodel_currentDomain);

		if (param.includeCurrentDomainAsKnowledgeExtraction) {
			topicModelListForKnowledgeExtraction = param.topicModelList_LastIteration;
		} else {
			// Knowledge is extracted from the domain other than the
			// current domain.
			topicModelListForKnowledgeExtraction = new ArrayList<TopicModel>();
			for (TopicModel topicModel : param.topicModelList_LastIteration) {
				if (!topicModel.corpus.domain.equals(param.domain)) {
					topicModelListForKnowledgeExtraction.add(topicModel);
				}
			}
			--param.minimumSupport;
		}
	}

	/**
	 * Title: LTM2
	 * Description: 
	 * @author: LiuFeng
	 * @date: 2017年3月24日 下午8:21:48
	 * @param topicmodel_currentDomain
	 */
	private void initializeFirstMarkovChainUsingExistingZ(TopicModel topicmodel_currentDomain) {
		
		// gamma
		gamma = topicmodel_currentDomain.getDocumentTopicDistrbution();
		
		// Phi
		phi =  topicmodel_currentDomain.getTopicWordDistributionOne();
		
		// Lambda
		lambda = topicmodel_currentDomain.getTopicWordDistribution();
		
	}

	/**
	 * Title: LTM2 Description:
	 * 
	 * @author: LiuFeng
	 * @param param2
	 * @param corpus2
	 * @date: 2017年3月24日 下午3:48:55
	 */
	private void initializeModel(Corpus corpus2, ModelParameters param2) {

		lambda = new double[param2.T][param2.V];
		gamma = new double[param2.D][param2.T];

		// initialize documents index array
		phi = new double[param2.D][param2.T][];
		for (int m = 0; m < param2.D; m++) {
			// Notice the limit of memory
			int N = corpus2.docs[m].length;
			for (int k = 0; k < param2.T; k++) {
				phi[m][k] = new double[N];
			}
		}

		Random rand = new Random();
		for (int m = 0; m < param2.D; m++) {
			int N = corpus2.docs[m].length;
			// System.out.println(N+"==N==");
			for (int k = 0; k < param2.T; k++) {
				gamma[m][k] = param2.alpha;
				for (int n = 0; n < N; n++) {
					double num = rand.nextDouble();
					phi[m][k][n] = num;
				}
			}
		}

		for (int k = 0; k < param2.T; k++) {

			for (int v = 0; v < param2.V; v++) {
				lambda[k][v] = param2.beta;
			}

		}
	}

	/**
	 * Create a new topic model with all variables initialized. The z[][] is
	 * assigned to the value loaded from other models.
	 */
	public LTMvb(Corpus corpus2, ModelParameters param2, double[][] dwdist, double[][][] twDistOne, double[][] twdist) {

		super(corpus2, param2);

		gamma = dwdist;

		lambda = twdist;

		phi = twDistOne;
		
		
	}

	@Override
	public void run() {
		
		ArrayList<PriorityQueue<Integer>> topWordIDList = getTopWordsUnderEachTopic(lambda);

		topicToMustlinksMap = getKnowledgeFromTopicModelResults(
				topWordIDList,
				topicModelListForKnowledgeExtraction, lambda,
				corpus.vocab);
		
		for (int i = 0; i < param.nBurnin; i++) {  //200
			inferParameters(i);
		}
		
	}

	private HashMap<Integer, MustLinks> getKnowledgeFromTopicModelResults(
			ArrayList<PriorityQueue<Integer>> topWordIDList,
			ArrayList<TopicModel> topicModelList_LastIteration,
			double[][] topicWordDistribution, Vocabulary vocab) {
		KnowledgeAsMustClustersGenerator knowledgeGenarator = new KnowledgeAsMustClustersGenerator(
				param);
		return knowledgeGenarator.generateKnowledgeAsMustLinks(topWordIDList,
				topicModelList_LastIteration, topicWordDistribution, vocab);
	}

	private void inferParameters(int i) {
		
		double[][] gammaTemp = new double[param.D][param.T];
		double[][] lambdaTemp = new double[param.T][param.V];
		double[][] initLambdaTemp = new double[param.T][param.V];
		double[] gammaTempSum = new double[param.D];
		double[] lambdaTempSum = new double[param.T];
		double[][] phiSum = new double[param.D][];
		
		for (int m = 0; m < param.D; m++) {
			for (int k = 0; k < param.T; k++) {
				gammaTemp[m][k] = param.alpha;
			}
		}

		for (int k = 0; k < param.T; k++) {
			for (int v = 0; v < param.V; v++) {
				lambdaTemp[k][v] = param.beta;
			}
		}
		
		for (int m = 0; m < param.D; m++) {
			int N = docs[m].length;
			for (int k = 0; k < param.T; k++) {
				for (int n = 0; n < N; n++) {
					int v = docs[m][n];
					gammaTemp[m][k] += phi[m][k][n];
					gammaTempSum[m] += phi[m][k][n];
					lambdaTemp[k][v] += phi[m][k][n];
					initLambdaTemp[k][v] += phi[m][k][n];
					lambdaTempSum[k] += phi[m][k][n];
				}
			}
		}		

		for (int m = 0; m < param.D; m++) {
			int N = docs[m].length;
			phiSum[m] = new double[N];
			for (int k = 0; k < param.T; k++) {
				for (int n = 0; n < N; n++) {
					int v = docs[m][n];
					double temp = digamma(gammaTemp[m][k]) + digamma(lambdaTemp[k][v]) - digamma(lambdaTempSum[k]);
					phi[m][k][n] = Math.exp(temp);
					phiSum[m][n] += phi[m][k][n];
				}
			}
		}

		for (int m = 0; m < param.D; m++) {
			for (int k = 0; k < param.T; k++) {
				gamma[m][k] = gammaTemp[m][k];
			}
		}

/******************************************************************************************/
		
		if (i > (param.nBurnin*0.8)) {
			ArrayList<PriorityQueue<Integer>> topWordIDList = getTopWordsUnderEachTopic(lambda);

			topicToMustlinksMap = getKnowledgeFromTopicModelResults(
					topWordIDList,
					topicModelListForKnowledgeExtraction, lambda,
					corpus.vocab);
			updateUrnMatrix(topicToMustlinksMap);
			
			
			HashMap<Integer, Map<Integer, Double>> knowledges = new HashMap<>();
			if (urn_Topic_W1_W2_Value!=null) {
				for (Integer topic : urn_Topic_W1_W2_Value.keySet()) {
					
					for (Integer w_i : urn_Topic_W1_W2_Value.get(topic).keySet()) {
						Map<Integer, Double> wjj = null;
						if (knowledges.containsKey(w_i)) {
							wjj = knowledges.get(w_i);
						} else {
							wjj = new HashMap<>();
							knowledges.put(w_i, wjj);
						}
						for (Integer w_j : urn_Topic_W1_W2_Value.get(topic).get(w_i).keySet()) {
							
							if (wjj.containsKey(w_j)) {
								wjj.put(w_j, wjj.get(w_j) + urn_Topic_W1_W2_Value.get(topic).get(w_i).get(w_j));
							} else {
								wjj.put(w_j, urn_Topic_W1_W2_Value.get(topic).get(w_i).get(w_j));
							}
						}
					}
				}
			}
			
			for (int k = 0; k < param.T; k++) {
				for (int v : knowledges.keySet()) {
					if (knowledges.get(v).size()>0) {
						for (int c = 0; c < 20; c++) {   //默认迭代20次
							double tempGrient = (trigamma(lambdaTemp[k][v]) - trigamma(lambdaTempSum[k])) * (initLambdaTemp[k][v]- lambdaTemp[k][v]) + grant(lambdaTemp,knowledges,v);
							lambdaTemp[k][v] = lambdaTemp[k][v] - 0.1 * tempGrient;
							lambdaTempSum[k] -= 0.1 * tempGrient;
						}
					}
				}
			}
			
		}
		
	
		
/******************************************************************************************/

		for (int k = 0; k < param.T; k++) {
			for (int v = 0; v < param.V; v++) {
				lambda[k][v] = lambdaTemp[k][v];
			}
		}
		
		for (int m = 0; m < param.D; m++) {
			int N = docs[m].length;
			for (int k = 0; k < param.T; k++) {
				for (int n = 0; n < N; n++) {
					phi[m][k][n] /= phiSum[m][n];
				}
			}
		}
	}

	@Override
	public double[][] getTopicWordDistribution() {
		return lambda;
	}

	@Override
	public double[][] getDocumentTopicDistrbution() {
		return gamma;
	}
	
	public static double GAMMA = 0.577215664901532860606512090082;
	public static double GAMMA_MINX = 1.e-12;
	public static double DIGAMMA_MINNEGX = -1250;
	public static double C_LIMIT = 49;
	public static double S_LIMIT = 1e-5;
	
	public static double digamma(double x) {
		if (x >= 0 && x < GAMMA_MINX) {
			x = GAMMA_MINX;
		}
		if (x < DIGAMMA_MINNEGX) {
			return digamma(DIGAMMA_MINNEGX + GAMMA_MINX);
		}
		if (x > 0 && x <= S_LIMIT) {
			return -GAMMA - 1 / x;
		}

		if (x >= C_LIMIT) {
			double inv = 1 / (x * x);
			return Math.log(x) - 0.5 / x - inv * ((1.0 / 12) + inv * (1.0 / 120 - inv / 252));
		}
		return digamma(x + 1) - 1 / x;
	}

	@Override
	public double[][][] getTopicWordDistributionOne() {
		
		return phi;
	}
	
	public void printKnowledge(String filepath) {
		StringBuilder sbKnowledge = new StringBuilder();
		for (int t = 0; t < param.T; ++t) {
			if (topicToMustlinksMap.containsKey(t)) {
				MustLinks mustlinks = topicToMustlinksMap.get(t);
				if (mustlinks.size() > 0) {
					sbKnowledge.append("<Topic=" + t + ">");
					sbKnowledge.append(System.getProperty("line.separator"));
					sbKnowledge.append(mustlinks);
					sbKnowledge.append("<\\Topic>");
					sbKnowledge.append(System.getProperty("line.separator"));
					sbKnowledge.append(System.getProperty("line.separator"));
				}
			}
		}
		FileReaderAndWriter.writeFile(filepath, sbKnowledge.toString());
	}
	
	/**
	 * Note that the counting matrixes will be updated in this function.
	 */
	private void updateUrnMatrix(HashMap<Integer, MustLinks> topicToMustlinksMap) {
		HashMap<Integer, HashMap<Integer, HashMap<Integer, Double>>> newurn_Topic_W1_W2_Value = new HashMap<Integer, HashMap<Integer, HashMap<Integer, Double>>>();
		for (int t = 0; t < param.T; ++t) {
			newurn_Topic_W1_W2_Value.put(t,
					new HashMap<Integer, HashMap<Integer, Double>>());
			HashMap<Integer, HashMap<Integer, Double>> urn_W1_W2_Value = newurn_Topic_W1_W2_Value
					.get(t);
			if (!topicToMustlinksMap.containsKey(t)) {
				// There is no knowledge under this topic.
				continue;
			}
			MustLinks mustlinks = topicToMustlinksMap.get(t);
			for (MustLink mustlink : mustlinks) {
				String wordstr1 = mustlink.wordpair.wordstr1;
				String wordstr2 = mustlink.wordpair.wordstr2;
				if (!corpus.vocab.containsWordstr(wordstr1)
						|| !corpus.vocab.containsWordstr(wordstr2)) {
					// The knowledge word does not appear in this domain.
					continue;
				}
				int w1 = corpus.vocab.getWordidByWordstr(wordstr1);
				int w2 = corpus.vocab.getWordidByWordstr(wordstr2);

				// We only need the off diagonal elements in the urn matrix
				// since the pair words in the mustlinks are different.
				if (w1 != w2) {
					// Use PMI to update urn matrix.
					int coDocFrequency = corpus.getCoDocumentFrequency(
							wordstr1, wordstr2) + 1;
					int docFrequency1 = corpus.getDocumentFrequency(wordstr1) + 1;
					int docFrequency2 = corpus.getDocumentFrequency(wordstr2) + 1;

					double Pxy = 1.0 * coDocFrequency / param.D;
					double Px = 1.0 * docFrequency1 / param.D;
					double Py = 1.0 * docFrequency2 / param.D;
					double PMI = Math.log(Pxy / (Px * Py));
					double gpuScale = param.pmiScaleToGPU * PMI;
					if (gpuScale <= 0) {
						continue;
					}
					// System.out.println(w1 + " " + w2 + " " + gpuScale);

					if (!urn_W1_W2_Value.containsKey(w1)) {
						urn_W1_W2_Value.put(w1, new HashMap<Integer, Double>());
					}
					HashMap<Integer, Double> urn_W2_Value = urn_W1_W2_Value
							.get(w1);
					urn_W2_Value.put(w2, gpuScale);

					if (!urn_W1_W2_Value.containsKey(w2)) {
						urn_W1_W2_Value.put(w2, new HashMap<Integer, Double>());
					}
					urn_W2_Value = urn_W1_W2_Value.get(w2);
					urn_W2_Value.put(w1, gpuScale);
				}
			}
		}
		urn_Topic_W1_W2_Value = newurn_Topic_W1_W2_Value;
	}
	
	private double grant(double[][] temp, Map<Integer, Map<Integer, Double>> knowCount, int v) {
		double sum = 0;
		
		for (int key : knowCount.get(v).keySet()) {
			
			double lambs = 0;
			double lambss = 0;
			for (int k = 0; k < param.T ; k++) {
				lambs += temp[k][v] * temp[k][key];	
				lambss += temp[k][v];
			}

			sum +=  knowCount.get(v).get(key) * ((1.0)/lambs) *  lambss;
		}
		return sum;
	}
	
	public static double trigamma(double x) {
		double p;
		int i;

		x = x + 6;
		p = 1 / (x * x);
		p = (((((0.075757575757576 * p - 0.033333333333333) * p + 0.0238095238095238) * p - 0.033333333333333) * p
				+ 0.166666666666667) * p + 1) / x + 0.5 * p;
		for (i = 0; i < 6; i++) {
			x = x - 1;
			p = 1 / (x * x) + p;
		}
		return (p);
	}
}

class TopicalWordComparator implements Comparator<Integer> {
	private double[] distribution = null;

	public TopicalWordComparator(double[] distribution2) {
		distribution = distribution2;
	}

	@Override
	public int compare(Integer w1, Integer w2) {
		if (distribution[w1] < distribution[w2]) {
			return -1;
		} else if (distribution[w1] > distribution[w2]) {
			return 1;
		}
		return 0;
	}
}
