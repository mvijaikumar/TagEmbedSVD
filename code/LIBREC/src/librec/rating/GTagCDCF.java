package librec.rating;

import librec.data.DenseMatrix;
import librec.data.MatrixEntry;
import librec.data.SparseMatrix;
import librec.intf.TagRecommender;

/**
 * Yue Shi, Martha Larson, Alan Hanjalic, <strong>Exploiting Social Tags for Cross-Domain Collaborative Filtering.</strong>,
 * arXiv 2013.
 * 
 * @author Vijaikumar Mylsamy
 * 
 */

public class GTagCDCF extends TagRecommender {

	private double lambda, alpha, beta;
	private DenseMatrix T;
	private SparseMatrix FU = null;
	private SparseMatrix FV = null;
	
	public GTagCDCF(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
		super(trainMatrix, testMatrix, fold);

		initByNorm = false;
	}

	public GTagCDCF(SparseMatrix trainMatrix, SparseMatrix testMatrix, SparseMatrix validMatrix, int fold)  throws Exception {
		super(trainMatrix, testMatrix, validMatrix, fold);
		
		initByNorm = false; 
		T = new DenseMatrix(numTags,numFactors);
		T.init();		
	}	
	
	@Override
	protected void initModel() throws Exception {
		super.initModel();
		
		FU = userTagMatrix;
		FV = itemTagMatrix;		
	}

	@Override
	protected void buildModel() throws Exception {
		
		double loc_lRate = lRate;
		lambda = regOptions.getDouble("-lambda", reg);		
		alpha  = regOptions.getDouble("-alpha", reg);
		beta   = regOptions.getDouble("-beta", reg); 
		
		for (int iter = 1; iter <= numIters; iter++) {
			
			//loc_lRate = loc_lRate * 0.98; lRate = loc_lRate;

			loss = 0;
			DenseMatrix PS = new DenseMatrix(numUsers, numFactors);
			DenseMatrix QS = new DenseMatrix(numItems, numFactors);
			DenseMatrix TS = new DenseMatrix(numTags, numFactors);

			for (MatrixEntry me : trainMatrix) {
				int u = me.row();
				int j = me.column();
				double ruj = me.get();

				double pred = P.row(u).inner(Q.row(j)); //predict(u, j, false);
				double euj = g(pred) - normalize(ruj);

				loss += euj * euj;

				for (int f = 0; f < numFactors; f++) {
					double puf = P.get(u, f);
					double qjf = Q.get(j, f);

					PS.add(u, f, euj * gd(pred) *  qjf );
					QS.add(j, f, euj * gd(pred) *  puf );
				}
			}			
			
			for (MatrixEntry me : FU) {
				int u = me.row();
				int t = me.column();
				double f_ut = me.get();

				if (f_ut <= 0)
					continue;
				
				double predf = P.row(u).inner(T.row(t));
				double eut   = g(predf) - normalizeTag(f_ut, true);				
				loss += alpha * eut * eut;
				
				for (int f = 0; f < numFactors; f++) {					
					PS.add(u, f, alpha * eut * gd(predf) * T.get(t, f));		
					TS.add(t, f, alpha * eut * gd(predf) * P.get(u, f));					
				}									
			}
			
			for (MatrixEntry me : FV) {
				int j = me.row();
				int t = me.column();
				double f_jt = me.get();

				if (f_jt <= 0)
					continue;
				
				double predf = Q.row(j).inner(T.row(t));
				double ejt = g(predf) - normalizeTag(f_jt, false);				
				loss += beta * ejt * ejt;
				
				for (int f = 0; f < numFactors; f++) {					
					QS.add(j, f, beta * ejt * gd(predf) * T.get(t, f));					
					TS.add(t, f, beta * ejt * gd(predf) * Q.get(j, f));
				}					
			}

			TS = TS.add(T.scale(lambda));
			QS = QS.add(Q.scale(lambda));
			PS = PS.add(P.scale(lambda));
			
			loss += P.fro_Norm();
			loss += Q.fro_Norm();
			loss += T.fro_Norm();			
			
			loss *= 0.5;
			
			P = P.add(PS.scale(-lRate));
			Q = Q.add(QS.scale(-lRate));
			T = T.add(TS.scale(-lRate));

			if (isConverged(iter))
				break;
			if( (iter>=5 )|| iter == numIters)			
				minmet = evalRatMetrics(bw, "  Dim : " +Integer.toString(numFactors) + "  lambda : " + Double.toString(lambda) + "  alpha : " + Double.toString(alpha) + "  beta : " + Double.toString(beta) + "  filename : " + cf.getString("trainpath") +"  ",minmet, iter==numIters);
		}		
	}

	@Override
	public double predict(int u, int j) { 
		double pred = 0; 
		if (u > trainMatrix.numRows()-1 || j > trainMatrix.numColumns()-1) {
			return globalMean;
		}	
		pred = DenseMatrix.rowMult(P, u, Q, j);
		return denormalize(g(pred));
	}
}
