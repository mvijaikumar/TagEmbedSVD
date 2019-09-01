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

package librec.rating;

import java.util.List;

import librec.data.DenseMatrix;
import librec.data.DenseVector;
import librec.data.MatrixEntry;
import librec.data.SparseMatrix;
import librec.intf.TagRecommender;

/**
 * Guo et al., <strong>TrustSVD: Collaborative Filtering with Both the Explicit and Implicit Influence of User Trust and
 * of Item Ratings</strong>, AAAI 2015.
 * 
 * @author guoguibing 
 * 
 */
public class TagTrustSVD extends TagRecommender {

	private DenseMatrix W, Y;
	private DenseVector reg_Uj_arr, reg_Tuin_arr, reg_Tu_arr;

	public TagTrustSVD(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
		super(trainMatrix, testMatrix, fold);
	}

	public TagTrustSVD(SparseMatrix trainMatrix, SparseMatrix testMatrix, SparseMatrix validMatrix, int fold)  throws Exception {
		super(trainMatrix, testMatrix, validMatrix, fold);

		initByNorm = false; //mine
		//initByNorm = false;
		
	}
	
	@Override
	protected void initModel() throws Exception {
		super.initModel();
		//##
		//int numUsers = trainMatrix.numRows();
		//int numItems = trainMatrix.numColumns();

		userBias = new DenseVector(numUsers);
		itemBias = new DenseVector(numItems);

		W = new DenseMatrix(numUsers, numFactors);
		Y = new DenseMatrix(numItems, numFactors);

		if (initByNorm) {
			userBias.init(initMean, initStd);
			itemBias.init(initMean, initStd);
			W.init(initMean, initStd);
			Y.init(initMean, initStd);
		} else {
			userBias.init();
			itemBias.init();
			W.init();
			Y.init();
		}

		reg_Tuin_arr = new DenseVector(numUsers);
		reg_Tu_arr = new DenseVector(numUsers);
		reg_Uj_arr = new DenseVector(numItems);

		userItemsCache = trainMatrix.rowColumnsCache(cacheSpec);
		userFriendsCache = userTagMatrix.rowColumnsCache(cacheSpec);

		for (int u = 0; u < numUsers; u++) {
			int count;
			if (u>trainMatrix.numRows()-1) //##
				count = 1;
			else
			    count = userTagMatrix.columnSize(u);
			reg_Tuin_arr.set(u, count > 0 ? 1.0 / Math.sqrt(count) : 1.0);
			if (u>trainMatrix.numRows()-1) //##
				count = 1;
			else
				count = userTagMatrix.rowSize(u);
			reg_Tu_arr.set(u, count > 0 ? 1.0 / Math.sqrt(count) : 1.0);
		}

		for (int j = 0; j < numItems; j++) {
			
			//##
			int count;
			if (j>trainMatrix.numColumns()-1)
				count = 1;
			else
				count = trainMatrix.columnSize(j);	
			
			reg_Uj_arr.set(j, count > 0 ? 1.0 / Math.sqrt(count) : 1.0);
		}
	}

	@Override
	protected void buildModel() throws Exception {
		
		double regS = regSU, mu = regOptions.getFloat("-mu", reg);; // ## mine
		double loc_lrate = lRate;
		for (int iter = 1; iter <= numIters; iter++) {
			loss = 0;
			lRate = loc_lrate;
			DenseMatrix PS = new DenseMatrix(numUsers, numFactors);
			DenseMatrix WS = new DenseMatrix(numUsers, numFactors);

			//System.out.print("\t Learning rate : " + lRate);
			for (MatrixEntry me : trainMatrix) {
				int u = me.row(); // user
				int j = me.column(); // item
				double ruj = me.get(); // rating

				// To speed up, directly access the prediction instead of invoking "pred = predict(u,j)"
				double bu = userBias.get(u);
				double bj = itemBias.get(j);
				double pred = globalMean + bu + bj + DenseMatrix.rowMult(P, u, Q, j);

				// Y
				List<Integer> rated_items = userItemsCache.get(u);
				if (rated_items.size() > 0) {
					double sum = 0;
					for (int i : rated_items)
						sum += DenseMatrix.rowMult(Y, i, Q, j);

					pred += sum / Math.sqrt(rated_items.size());
				}
				
				List<Integer> trusted_users = userFriendsCache.get(u);
				/*// W				
				if (trusted_users.size() > 0) {
					double sum = 0.0;
					for (int v : trusted_users)
						sum += DenseMatrix.rowMult(W, v, Q, j);

					pred += sum / Math.sqrt(trusted_users.size());
				}*/ //1

				double euj = pred - ruj;
				//euj = mu * euj;
				//loss += euj * euj;
				
				
				///////////////////////////////////////////////////
				if(Integer.parseInt(rateDao.getUserId(u)) <= 2000) { //##
					if (mu>0) {
						euj =  mu * (pred-ruj);
						loss += euj * euj/ mu;
					} else {
						euj =  0;
						loss += 0;
					}
					
				} else {
					euj =  pred-ruj;
					loss += euj * euj;
				}
				///////////////////////////////////////////////////////

				double rated_items_size = Math.sqrt(rated_items.size());
				double trusted_users_size = Math.sqrt(trusted_users.size());

				// update factors
				double reg_Iu = 1.0 / rated_items_size;
				double reg_Uj = reg_Uj_arr.get(j);

				//double sgd = euj + regB * reg_Iu * bu;
				double sgd = euj + regB * bu; // ##AA
				userBias.add(u, -lRate * sgd);

				//sgd = euj + regB * reg_Uj * bj;
				sgd = euj + regB * bj; //##AA
				itemBias.add(j, -lRate * sgd);

				//loss += regB * reg_Iu * bu * bu;
				loss += regB * bu * bu;//##AA
				//loss += regB * reg_Uj * bj * bj;
				loss += regB * bj * bj; //## AA

				double[] sum_ys = new double[numFactors]; 
				for (int f = 0; f < numFactors; f++) {
					double sum = 0;
					for (int i : rated_items)
						sum += Y.get(i, f);

					sum_ys[f] = rated_items_size > 0 ? sum / rated_items_size : sum;
				}

				/*double[] sum_ts = new double[numFactors];
				for (int f = 0; f < numFactors; f++) {
					double sum = 0;
					for (int v : trusted_users)
						sum += W.get(v, f);

					sum_ts[f] = trusted_users_size > 0 ? sum / trusted_users_size : sum;
				}*/ //2

				for (int f = 0; f < numFactors; f++) {
					double puf = P.get(u, f);
					double qjf = Q.get(j, f);

					//double delta_u = euj * qjf + regU * reg_Iu * puf;
					double delta_u = euj * qjf + regU * puf; //##AA
					//double delta_j = euj * (puf + sum_ys[f] + sum_ts[f]) + regI * reg_Uj * qjf; //3
					//double delta_j = euj * (puf + sum_ys[f]) + regI * reg_Uj * qjf; //## deleted respetive w term
					double delta_j = euj * (puf + sum_ys[f]) + regI * qjf; //##AA
					PS.add(u, f, delta_u);
					Q.add(j, f, -lRate * delta_j);

					//loss += regU * reg_Iu * puf * puf + regI * reg_Uj * qjf * qjf;
					loss += regU * puf * puf + regI * qjf * qjf; //##AA AA

					for (int i : rated_items) {
						double yif = Y.get(i, f);

						double reg_Ui = reg_Uj_arr.get(i);
						//double delta_y = euj * qjf / rated_items_size + regI * reg_Ui * yif;
						double delta_y = euj * qjf / rated_items_size + regI * yif; //##AA
						Y.add(i, f, -lRate * delta_y);

						//loss += regI * reg_Ui * yif * yif;
						loss += regI * yif * yif; //##AA
					}

					/* // update wvf
					for (int v : trusted_users) {
						double wvf = W.get(v, f);

						double reg_Tuin = reg_Tuin_arr.get(v);
						//double delta_t = euj * qjf / trusted_users_size + regU * reg_Tuin * wvf; 
						double delta_t = euj * qjf / trusted_users_size + regU * wvf; //AA##
						WS.add(v, f, delta_t);

						//loss += regU * reg_Tuin * wvf * wvf;
						loss += regU * wvf * wvf; //##AA
					}*/ //4
				}
			}

			for (MatrixEntry me : userTagMatrix) {
				int u = me.row();
				int v = me.column();
				//double tuv = 5 * me.get(); //##2
				double tuv = me.get();
				if (tuv == 0)
					continue;

				//System.out.println("similarity value : " +tuv);
				double pred = DenseMatrix.rowMult(P, u, W, v);
				double eut = pred - tuv;

				loss += regS * eut * eut;

				double csgd = regS * eut;
				double reg_Tu = reg_Tu_arr.get(u);

				for (int f = 0; f < numFactors; f++) {
					double puf = P.get(u, f);
					double wvf = W.get(v, f);

					//PS.add(u, f, csgd * wvf + regS * reg_Tu * puf);
					PS.add(u, f, csgd * wvf + regS * puf); //##AA
					WS.add(v, f, csgd * puf);

					//loss += regS * reg_Tu * puf * puf;
					loss += regS * puf * puf; //##AA
				}
			}

			P = P.add(PS.scale(-lRate));
			W = W.add(WS.scale(-lRate));

			loss *= 0.5;

			if (isConverged(iter))
				break;
			if (iter>3)			
				minmet = evalRatMetrics(bw, "  Dim : " + Float.toString(numFactors) + "  regU : " + Float.toString(regU) + "  regI : " + Float.toString(regI) + "  regB : " + Float.toString(regB) + "  simU : " + Float.toString(regSU) + "  simV : " + Float.toString(regSV) + "  mu : " + Double.toString(mu) + "  filename : " + cf.getString("trainpath") +"  ",minmet, iter==numIters); //mine            

		}
		System.out.println("\nregU : "+regU +"  regI : "+ regI+ "  regSU : " + regSU +"  regSV : " + regSV + "  mu  : " + mu);
		System.out.println("res file : " + resFile);

		//}// end of training
	}

	@Override
	public double predict(int u, int j) throws Exception {
		double pred = globalMean + userBias.get(u) + itemBias.get(j) + DenseMatrix.rowMult(P, u, Q, j);

		// Y
		List<Integer> rated_items = userItemsCache.get(u);
		if (rated_items.size() > 0) {
			double sum = 0;
			for (int i : rated_items)
				sum += DenseMatrix.rowMult(Y, i, Q, j);

			pred += sum / Math.sqrt(rated_items.size());
		}

		/*// W
		List<Integer> trusted_users = userFriendsCache.get(u);
		if (trusted_users.size() > 0) {
			double sum = 0.0;
			for (int v : trusted_users)
				sum += DenseMatrix.rowMult(W, v, Q, j);

			pred += sum / Math.sqrt(trusted_users.size());
		}*/ //5

		return pred;
	}
}