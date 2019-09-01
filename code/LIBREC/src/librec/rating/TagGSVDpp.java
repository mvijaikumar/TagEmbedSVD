package librec.rating;

import java.io.IOException;
import java.util.List;

import librec.data.DenseMatrix;
import librec.data.DenseVector;
import librec.data.MatrixEntry;
import librec.data.SparseMatrix;
import librec.intf.TagRecommender;


public class TagGSVDpp extends TagRecommender {

	private double lambda;
	private DenseMatrix X, Y;
	private DenseVector reg_Tu_vec, reg_Tj_vec;

	public TagGSVDpp(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
		super(trainMatrix, testMatrix, fold);

		initByNorm = false;
	}
	
	public TagGSVDpp(SparseMatrix trainMatrix, SparseMatrix testMatrix, SparseMatrix validMatrix, int fold) throws IOException {
		super(trainMatrix, testMatrix, validMatrix, fold);

		initByNorm = false;
	}
		
	@Override
	protected void initModel() throws Exception {
		super.initModel();

		userTagCache = userTagMatrix.rowColumnsCache(cacheSpec);
		itemTagCache = itemTagMatrix.rowColumnsCache(cacheSpec);
		
		X = new DenseMatrix(numTags,numFactors);
		Y = new DenseMatrix(numTags,numFactors);
		if (initByNorm) {
			X.init(initMean, initStd);
			Y.init(initMean, initStd);
		} else {		
			X.init();
			Y.init();
		}
		reg_Tu_vec = new DenseVector(numUsers); // Set of tags used by user u 
		reg_Tj_vec = new DenseVector(numItems); // Set of tags used on the item j

		userTagCache   = userTagMatrix.rowColumnsCache(cacheSpec);
		itemTagCache   = itemTagMatrix.rowColumnsCache(cacheSpec);

		for (int u = 0; u < numUsers; u++) {
			int freq_count = 0;
			List<Integer>  Tu = userTagCache.get(u);
			for (int t : Tu) 
				freq_count += userTagMatrix.get(u,t);
			reg_Tu_vec.set(u, freq_count > 0 ? 1.0 / freq_count : 1.0);
		}
		for (int j = 0; j < numItems; j++) {
			int freq_count = 0;
			List<Integer>  Tj = itemTagCache.get(j);
			for (int t : Tj) 
				freq_count += itemTagMatrix.get(j,t);
			reg_Tj_vec.set(j, freq_count > 0 ? 1.0 / freq_count : 1.0);
		}		
	}
	
	@Override
	protected void buildModel() throws Exception { 
		
		double loc_lrate = lRate;
		lambda = regOptions.getDouble("-lambda", reg);
		for (int iter = 1; iter <= numIters; iter++) {
			loss = 0; 
			//lRate = loc_lrate; loc_lrate = loc_lrate*.98;

			for (MatrixEntry me : trainMatrix) {
				int u = me.row(); // user
				int j = me.column(); // item
				double ruj = me.get(); // rating
		
				double reg_Tu = reg_Tu_vec.get(u);
				double reg_Tj = reg_Tj_vec.get(j);
				
				double pred = predict(u,j);
				double euj = pred - ruj;	
				loss += euj * euj;
				
				List<Integer> user_tags = userTagCache.get(u);
				List<Integer> item_tags = itemTagCache.get(j);
				
				//X
				double[] sum_xs = new double[numFactors];
				for (int f = 0; f < numFactors; f++) {
					double sum = 0;
					for (int s : user_tags)
						sum += userTagMatrix.get(u,s) * X.get(s, f);

					sum_xs[f] = sum * reg_Tu;
				}
				
				// Y
				double[] sum_yt = new double[numFactors];
				for (int f = 0; f < numFactors; f++) {
					double sum = 0;
					for (int t : item_tags)
						sum += itemTagMatrix.get(j, t) * Y.get(t, f);

					sum_yt[f] = sum * reg_Tj;
				}
				
				for (int f = 0; f < numFactors; f++) {
					double puf = P.get(u, f);
					double qjf = Q.get(j, f);
					
					double delta_u = euj * (qjf + sum_yt[f]) + lambda *  puf;
					double delta_j = euj * (puf + sum_xs[f]) + lambda *  qjf;
					
					for (int s : user_tags) {
						double xsf = X.get(s, f);
						double delta_x = euj * userTagMatrix.get(u, s) * reg_Tu_vec.get(u) * (qjf + sum_yt[f]) + lambda * xsf;
						
						X.add(u, f, -lRate * delta_x);						
						loss += lambda * xsf * xsf;
					}
					
					for (int t : item_tags) {
						double ytf = Y.get(t, f);
						double delta_y = euj * itemTagMatrix.get(j, t) * reg_Tj_vec.get(j) * (puf + sum_xs[f]) + lambda * ytf;
						
						Y.add(j, f, -lRate * delta_y);						
						loss += lambda * ytf * ytf;
					}					
					P.add(u, f, -lRate * delta_u);
					Q.add(j, f, -lRate * delta_j);					
				}				
			}
			
			loss *= 0.5;
			if (isConverged(iter))
				break;
			if ((iter>=3) || iter == numIters)
				minmet = evalRatMetrics(bw, "  Dim : " + Float.toString(numFactors) + "  lambda : " + Double.toString(lambda) + "  filename : " + cf.getString("trainpath") +"  ",minmet, iter==numIters);
		}
	}
	
	@Override
	public double predict(int u, int j) throws Exception {
		
		
		if ((u>trainMatrix.numRows()-1) || j>trainMatrix.numColumns()-1)
			return globalMean;		
		
		double pred = DenseMatrix.rowMult(P, u, Q, j);
		
		
		List<Integer> user_tags = userTagCache.get(u);
		List<Integer> item_tags = itemTagCache.get(j);
		double reg_Tu = reg_Tu_vec.get(u);
		double reg_Tj = reg_Tj_vec.get(j);
		
		// X
		double[] sum_xs = new double[numFactors];
		for (int f = 0; f < numFactors; f++) {
			double sum = 0;
			for (int s : user_tags)
				sum += userTagMatrix.get(u,s) * X.get(s, f);

			sum_xs[f] = sum * reg_Tu;
		}
		
		// Y
		double[] sum_yt = new double[numFactors];
		for (int f = 0; f < numFactors; f++) {
			double sum = 0;
			for (int t : item_tags)
				sum += itemTagMatrix.get(j, t) * Y.get(t, f);

			sum_yt[f] = sum * reg_Tj;
		}
		
		for (int f = 0; f < numFactors; f++) {
			pred += P.get(u,f) * sum_yt[f] + Q.get(j,f) * sum_xs[f] + sum_xs[f] * sum_yt[f];
		}
		return pred;
	}
}
