// Copyright (C) 2014 Guibing Guo
//
// This file is part of LibRec.
//
// LibRec is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// LibRec is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with LibRec. If not, see <http://www.gnu.org/licenses/>.
//

package librec.intf;

import librec.data.Configuration;
import librec.data.DenseMatrix;
import librec.data.DenseVector;
import librec.data.SparseMatrix;
import librec.util.FileIO;
import librec.util.LineConfiger;
import librec.util.Logs;
import librec.util.Strings;

/**
 * Recommenders using iterative learning techniques
 * 
 * @author guoguibing
 * 
 */
@Configuration("factors, lRate, maxLRate, regB, regU, regI, iters, boldDriver")
public abstract class IterativeRecommender extends Recommender {

	/************************************ Static parameters for all recommenders ***********************************/
	// init, maximum learning rate, momentum
	protected static float initLRate, maxLRate, momentum;
	// line configer for regularization parameters
	protected static LineConfiger regOptions;
	// user, item and bias regularization
	protected static float regU, regI, regB, reg;
	// number of factors
	protected static int numFactors;
	// number of iterations
	protected static int numIters;


	// whether to adjust learning rate automatically
	protected static boolean isBoldDriver;
	// decay of learning rate
	protected static float decay;

	// indicator of static field initialization
	public static boolean resetStatics = true;

	/************************************ Recommender-specific parameters ****************************************/
	// factorized user-factor matrix
	protected DenseMatrix P;

	// factorized item-factor matrix
	protected DenseMatrix Q;

	// user biases
	protected DenseVector userBias;
	// item biases
	protected DenseVector itemBias;

	// adaptive learn rate
	protected double lRate;
	// objective loss
	protected double loss, last_loss = 0;
	// predictive measure
	protected double measure, last_measure = 0;

	// initial models using normal distribution
	protected boolean initByNorm;

	
	public IterativeRecommender(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
		super(trainMatrix, testMatrix, fold);
		

		// initialization 
		if (resetStatics) {
			resetStatics = false;

			LineConfiger lc = cf.getParamOptions("learn.rate");
			if (lc != null) {
				initLRate = Float.parseFloat(lc.getMainParam());
				maxLRate = lc.getFloat("-max", -1);
				isBoldDriver = lc.contains("-bold-driver");
				decay = lc.getFloat("-decay", -1);
				momentum = lc.getFloat("-momentum", 50);
			}

			regOptions = cf.getParamOptions("reg.lambda");
			if (regOptions != null) {
				reg = Float.parseFloat(regOptions.getMainParam());
				regU = regOptions.getFloat("-u", reg);
				regI = regOptions.getFloat("-i", reg);
				regB = regOptions.getFloat("-b", reg);
			}

			numFactors = cf.getInt("num.factors", 10);
			numIters = cf.getInt("num.max.iter", 100);

		}

		// method-specific settings
		lRate = initLRate;
		initByNorm = true;
	}

	
	
	public IterativeRecommender(SparseMatrix trainMatrix, SparseMatrix testMatrix, SparseMatrix validMatrix, int fold){
		//super(trainMatrix, testMatrix, fold);
		super(trainMatrix, testMatrix, validMatrix, fold);

		// initialization 
		if (resetStatics) {
			resetStatics = false;

			LineConfiger lc = cf.getParamOptions("learn.rate");
			if (lc != null) {
				initLRate = Float.parseFloat(lc.getMainParam());
				maxLRate = lc.getFloat("-max", -1);
				isBoldDriver = lc.contains("-bold-driver");
				decay = lc.getFloat("-decay", -1);
				momentum = lc.getFloat("-momentum", 50);
			}

			regOptions = cf.getParamOptions("reg.lambda");
			if (regOptions != null) {
				reg = Float.parseFloat(regOptions.getMainParam());
				regU = regOptions.getFloat("-u", reg);
				regI = regOptions.getFloat("-i", reg);
				regB = regOptions.getFloat("-b", reg);
			}

			numFactors = cf.getInt("num.factors", 10);
			numIters = cf.getInt("num.max.iter", 100);
			resFile = cf.getString("res.file", "./resFile.txt");
			System.out.println(resFile);
		}

		// method-specific settings
		lRate = initLRate;
		initByNorm = true;
	}

	/**
	 * default prediction method
	 */
	@Override
	public double predict(int u, int j) throws Exception {
		if ((u>trainMatrix.numRows()-1) || j>trainMatrix.numColumns()-1)
			return globalMean;
		
		return DenseMatrix.rowMult(P, u, Q, j);
	}

	/**
	 * Post each iteration, we do things:
	 * 
	 * <ol>
	 * <li>print debug information</li>
	 * <li>check if converged</li>
	 * <li>if not, adjust learning rate</li>
	 * </ol>
	 * 
	 * @param iter
	 *            current iteration
	 * 
	 * @return boolean: true if it is converged; false otherwise
	 * 2016/8/2 RB I don't know if this is correct or not. The logic of the original is preserved but
	 * I don't understand it that well.
	 * If we are using ranking metrics, then use loss as the early stop measure.
	 * If not, use whatever metric is specified. I am restricting it to rating metrics -- diversity and time
	 * don't make sense, but I'm not sure what a different ranking metric can't be specific for early stop.
	 * 
	 */
	protected boolean isConverged(int iter) throws Exception {

		float delta_loss = (float) (last_loss - loss);

		if (earlyStopMeasure != null) {
			if (measures.hasRankingMetrics()){
				earlyStopMeasure = "Loss";
			}

			if (earlyStopMeasure.equals("Loss")) {
				measure = loss;
				last_measure = last_loss;
			} else {
				boolean flag = isResultsOut;
				isResultsOut = false; // to stop outputs
				measure = measures.getRatingMetric(earlyStopMeasure).getValue();
				isResultsOut = flag; // recover the flag
			}
		}

		float delta_measure = (float) (last_measure - measure);

		// print out debug info
		if (verbose) {
			String learnRate = lRate > 0 ? ", learn_rate = " + (float) lRate : "";

			String earlyStop = "";
			if (earlyStopMeasure != null && !earlyStopMeasure.equals("Loss")) {
				earlyStop = String.format(", %s = %.6f, delta_%s = %.6f", new Object[] { earlyStopMeasure,
						(float) measure, earlyStopMeasure, delta_measure });
			}

			Logs.debug("{}{} iter {}: loss = {}, delta_loss = {}{}{}", new Object[] { algoName, foldInfo, iter,
					(float) loss, delta_loss, earlyStop, learnRate });
		}

		if (Double.isNaN(loss) || Double.isInfinite(loss)) {
			Logs.error("Loss = NaN or Infinity: current settings does not fit the recommender! Change the settings and try again!");
			System.exit(-1);
		}

		// check if converged
		boolean cond1 = Math.abs(loss) < 1e-5;
		boolean cond2 = (delta_measure > 0) && (delta_measure < 1e-5);
		boolean converged = cond1 || cond2;

		// if not converged, update learning rate
		if (!converged)
			updateLRate(iter);

		last_loss = loss;
		last_measure = measure;

		return converged;
	}

	/**
	 * Update current learning rate after each epoch <br/>
	 * 
	 * <ol>
	 * <li>bold driver: Gemulla et al., Large-scale matrix factorization with distributed stochastic gradient descent,
	 * KDD 2011.</li>
	 * <li>constant decay: Niu et al, Hogwild!: A lock-free approach to parallelizing stochastic gradient descent, NIPS
	 * 2011.</li>
	 * <li>Leon Bottou, Stochastic Gradient Descent Tricks</li>
	 * <li>more ways to adapt learning rate can refer to: http://www.willamette.edu/~gorr/classes/cs449/momrate.html</li>
	 * </ol>
	 * 
	 * @param iter
	 *            the current iteration
	 */
	protected void updateLRate(int iter) {

		if (lRate <= 0)
			return;

		if (isBoldDriver && iter > 1)
			lRate = Math.abs(last_loss) > Math.abs(loss) ? lRate * 1.05 : lRate * 0.5;
		else if (decay > 0 && decay < 1)
			lRate *= decay;

		// limit to max-learn-rate after update
		if (maxLRate > 0 && lRate > maxLRate)
			lRate = maxLRate;
	}

	@Override
	protected void initModel() throws Exception {

		P = new DenseMatrix(numUsers, numFactors);
		Q = new DenseMatrix(numItems, numFactors);

		// initialize model
		if (initByNorm) {
			P.init(initMean, initStd);
			Q.init(initMean, initStd);
		} else {
			P.init(); // P.init(smallValue);
			Q.init(); // Q.init(smallValue);
		}

	}

	protected void saveModel() throws Exception {
		// make a folder
		String dirPath = FileIO.makeDirectory(tempDirPath, algoName);

		// suffix info
		String suffix = foldInfo + ".bin";

		// writing training, test data
		FileIO.serialize(trainMatrix, dirPath + "trainMatrix" + suffix);
		FileIO.serialize(testMatrix, dirPath + "testMatrix" + suffix);

		// write matrices P, Q
		FileIO.serialize(P, dirPath + "userFactors" + suffix);
		FileIO.serialize(Q, dirPath + "itemFactors" + suffix);

		// write vectors
		if (userBias != null)
			FileIO.serialize(userBias, dirPath + "userBiases" + suffix);
		if (itemBias != null)
			FileIO.serialize(itemBias, dirPath + "itemBiases" + suffix);

		Logs.debug("Learned models are saved to folder \"{}\"", dirPath);
	}

	protected void loadModel() throws Exception {
		// make a folder
		String dirPath = FileIO.makeDirectory(tempDirPath, algoName);

		Logs.debug("A recommender model is loaded from {}", dirPath);

		// suffix info
		String suffix = foldInfo + ".bin";

		trainMatrix = (SparseMatrix) FileIO.deserialize(dirPath + "trainMatrix" + suffix);
		testMatrix = (SparseMatrix) FileIO.deserialize(dirPath + "testMatrix" + suffix);

		// write matrices P, Q
		P = (DenseMatrix) FileIO.deserialize(dirPath + "userFactors" + suffix);
		Q = (DenseMatrix) FileIO.deserialize(dirPath + "itemFactors" + suffix);

		// write vectors
		userBias = (DenseVector) FileIO.deserialize(dirPath + "userBiases" + suffix);
		itemBias = (DenseVector) FileIO.deserialize(dirPath + "itemBiases" + suffix);
	}

	@Override
	public String toString() {
		return Strings.toString(new Object[] { numFactors, initLRate, maxLRate, regB, regU, regI, numIters,
				isBoldDriver });
	}

}
