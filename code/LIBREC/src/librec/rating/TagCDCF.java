package librec.rating;

import librec.data.DenseMatrix;
import librec.data.MatrixEntry;
import librec.data.SparseMatrix;
import librec.intf.TagRecommender;

/**
 * Yue Shi, Martha Larson, Alan Hanjalic, <strong>Tags as Bridges between Domains: Improving Recommendation with Tag-Induced Cross-Domain Collaborative Filtering.</strong>,
 * UMAP 2011.
 * 
 * @author Vijaikumar Mylsamy
 * 
 */

public class TagCDCF extends TagRecommender {

	private SparseMatrix SU = null;
	private SparseMatrix SV = null;
	private double alpha, beta, lambda;
	
	public TagCDCF(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
		super(trainMatrix, testMatrix, fold);

		initByNorm = false;
	}

	public TagCDCF(SparseMatrix trainMatrix, SparseMatrix testMatrix, SparseMatrix validMatrix, int fold)  throws Exception {
		super(trainMatrix, testMatrix, validMatrix, fold);
		
		initByNorm = false; 	
	}	
	
	@Override
	protected void initModel() throws Exception {
		super.initModel();

		SU = userTagMatrix;
		SV = itemTagMatrix;		
	}

	@Override
	protected void buildModel() throws Exception {
		
		double loc_lRate = lRate;
		lambda = regOptions.getDouble("-lambda", reg);		
		alpha  = regOptions.getDouble("-alpha", reg);
		beta   = regOptions.getDouble("-beta", reg); 
		
		System.out.println("alpha : " + alpha + " beta : " + beta  + " lam : " + lambda);
		
		for (int iter = 1; iter <= numIters; iter++) {
			
			loc_lRate = loc_lRate * 1;
			lRate = loc_lRate;

			loss = 0;
			DenseMatrix PS = new DenseMatrix(numUsers, numFactors);
			DenseMatrix QS = new DenseMatrix(numItems, numFactors);

			for (MatrixEntry me : trainMatrix) {
				int u = me.row();
				int j = me.column();
				double ruj = me.get();

				double pred = P.row(u).inner(Q.row(j));
				double euj = pred - ruj;
				loss += euj * euj;

				for (int f = 0; f < numFactors; f++) {
					double puf = P.get(u, f);
					double qjf = Q.get(j, f);

					PS.add(u, f, euj * qjf );
					QS.add(j, f, euj * puf );
				}
			}			
			
			for (MatrixEntry me : SU) {
				int u = me.row();
				int t = me.column();
				double s_ut = 5 * me.get(); //##

				if (s_ut <= 0)
					continue;
				
				double preds = P.row(u).inner(P.row(t));
				double eut   = preds - s_ut;				
				loss += alpha * eut * eut;
				
				for (int f = 0; f < numFactors; f++) {					
					PS.add(u, f, alpha * eut *  P.get(t, f));		
					PS.add(t, f, alpha * eut *  P.get(u, f));					
				}									
			}
			
			for (MatrixEntry me : SV) {
				int j = me.row();
				int t = me.column();
				double s_jt = 5 * me.get(); //##

				if (s_jt <= 0)
					continue;
				
				double preds = Q.row(j).inner(Q.row(t));
				double ejt = preds - s_jt;				
				loss += beta * ejt * ejt;
				
				for (int f = 0; f < numFactors; f++) {					
					QS.add(j, f, beta * ejt * Q.get(t, f));					
					QS.add(t, f, beta * ejt * Q.get(j, f));
				}					
			}

			QS = QS.add(Q.scale(lambda));
			PS = PS.add(P.scale(lambda));
			
			loss += P.fro_Norm();
			loss += Q.fro_Norm();
			
			loss *= 0.5;
			
			P = P.add(PS.scale(-lRate));
			Q = Q.add(QS.scale(-lRate));

			if (isConverged(iter))
				break;
			if (iter>=3 ||  iter == numIters)		
				minmet = evalRatMetrics(bw, "  Dim : " +Integer.toString(numFactors) + "  lambda : " + Double.toString(lambda) + "  alpha : " + Double.toString(alpha) + "  beta : " + Double.toString(beta) + "  filename : " + cf.getString("trainpath") +"  ",minmet, iter==numIters);
		}		
	}

	@Override
	public double predict(int u, int j) {
		
		if ((u>trainMatrix.numRows()-1) || j>trainMatrix.numColumns()-1)
			return globalMean;
		
		double pred = DenseMatrix.rowMult(P, u, Q, j);
		return pred;
	}
}
