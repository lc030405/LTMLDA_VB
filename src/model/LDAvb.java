package model;

import java.util.Random;

import nlp.Corpus;

/**
 * This implements the LDA model (Blei et al., 2003).
 */
public class LDAvb extends TopicModel {

	double[][] lambda;// Variational Parameters for topic-word distribution K*V
	double[][] gamma;// Variational Parameters for doc-topic distribution M*K
	double[][][] phi;// Variational Parameters for doc-word-topic distribution

	/**
	 * Create a new topic model with all variables initialized. The z[][] is
	 * randomly assigned.
	 */
	public LDAvb(Corpus corpus2, ModelParameters param2) {
		super(corpus2, param2);

		initializeModel(corpus2, param2);
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

		// initialize topic lable z for each word
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
	public LDAvb(Corpus corpus2, ModelParameters param2, double[][] dwdist, double[][][] twDistOne, double[][] twdist) {

		super(corpus2, param2);

		gamma = dwdist;

		lambda = twdist;

		phi = twDistOne;
	}

	@Override
	public void run() {
		for (int i = 0; i < param.nBurnin; i++) {  //200
			inferParameters();
		}
		
	}

	private void inferParameters() {
		
		double[][] gammaTemp = new double[param.D][param.T];
		double[][] lambdaTemp = new double[param.T][param.V];
//		double[][] initLambdaTemp = new double[param.T][param.V];
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
//					initLambdaTemp[k][v] += phi[m][k][n];
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
}
