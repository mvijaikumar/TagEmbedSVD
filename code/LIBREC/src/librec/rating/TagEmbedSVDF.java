package librec.rating;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.Hashtable;
import java.util.List;

import librec.data.DenseMatrix;
import librec.data.DenseVector;
import librec.data.MatrixEntry;
import librec.data.SparseMatrix;
import librec.intf.TagRecommender;

public class TagEmbedSVDF extends TagRecommender {

	private DenseMatrix W, X, Y, E, F, sumUserTag, sumItemTag;
	private DenseVector reg_Uj_vec, reg_Ut_vec, reg_Iu_vec, reg_Tu_vec, reg_It_vec, reg_Tj_vec;
	private double alpha, beta, regM, mu, loc_lrate;
	private DenseVector sumEt, sumFt;
	private Hashtable<String, DenseVector> tagtoembedding;
	int prediter = 0;
	
	public TagEmbedSVDF(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
		super(trainMatrix, testMatrix, fold);
	}

	public TagEmbedSVDF(SparseMatrix trainMatrix, SparseMatrix testMatrix, SparseMatrix validMatrix, int fold)  throws Exception {
		super(trainMatrix, testMatrix, validMatrix, fold);
		initByNorm = false;		
	}
	
	@Override
	protected void initModel() throws Exception {
		super.initModel();

		userBias = new DenseVector(numUsers);
		itemBias = new DenseVector(numItems);
		
		Y = new DenseMatrix(numItems, numFactors);
		W = new DenseMatrix(numUsers, numFactors);
		X = new DenseMatrix(numItems, numFactors);
		E = new DenseMatrix(numFactors, 300);
		F = new DenseMatrix(numFactors, 300);
		
		if (initByNorm) {
			userBias.init(initMean, initStd);
			itemBias.init(initMean, initStd);			
			Y.init(initMean, initStd);
			W.init(initMean, initStd);
			X.init(initMean, initStd);
			E.init(initMean,initStd);
			F.init(initMean,initStd);
		} else {
			userBias.init();
			itemBias.init();			
			Y.init();
			W.init();
			X.init();
			E.init();
			F.init();
		}
		// initializing sumEt, sumFt
		sumEt = new DenseVector(numFactors);
		sumFt = new DenseVector(numFactors);
		
		reg_Iu_vec = new DenseVector(numUsers); // Set of items rated by user u
		reg_Uj_vec = new DenseVector(numItems); // Set of users who rated items j		
		reg_Tu_vec = new DenseVector(numUsers); // Set of tags used by user u 
		reg_Ut_vec = new DenseVector(numTags);  // Set of users who used the tag t	
		reg_Tj_vec = new DenseVector(numItems); // Set of tags used on the item j
		reg_It_vec = new DenseVector(numTags);  // Set of items tagged by tag t	

		userItemsCache = trainMatrix.rowColumnsCache(cacheSpec);
		userTagCache   = userTagMatrix.rowColumnsCache(cacheSpec);
		itemTagCache   = itemTagMatrix.rowColumnsCache(cacheSpec);

		for (int u = 0; u < numUsers; u++) {
			double count = u > (trainMatrix.numRows()-1) ? 0 : trainMatrix.rowSize(u) ;			
			reg_Iu_vec.set(u, count > 0 ? 1.0 / Math.sqrt(count) : 1.0);			
			
			count = 0;
			List<Integer> user_tags = userTagCache.get(u);
			for (int k : user_tags) {
				double tuk = userTagMatrix.get(u,k);
				count = count + tuk;
			}
			count = u > (userTagMatrix.numRows()-1) ? 0 : count;			
			reg_Tu_vec.set(u, count > 0 ? 1.0 / Math.sqrt(count) : 1.0);
		}

		for (int j = 0; j < numItems; j++) {
			double count = j > (trainMatrix.numColumns()-1) ? 0 : trainMatrix.columnSize(j) ;			
			reg_Uj_vec.set(j, count > 0 ? 1.0 / Math.sqrt(count) : 1.0);	
			
			count = 0;
			List<Integer> item_tags = itemTagCache.get(j);
			for (int k : item_tags) {
				double tjk = itemTagMatrix.get(j,k);
				count = count + tjk ;
			}
			count = j > (itemTagMatrix.numRows()-1) ? 0 : itemTagMatrix.rowSize(j);			
			reg_Tj_vec.set(j, count > 0 ? 1.0 / Math.sqrt(count) : 1.0);
		}
		
		/* // Modify as in previous loops according to frequency
		for (int t = 0; t < numTags; t++) {
			double count = t > (userTagMatrix.numColumns()-1) ? 0 : userTagMatrix.columnSize(t) ;			
			reg_Ut_vec.set(t, count > 0 ? 1.0 / Math.sqrt(count) : 1.0);
			
			count = t > (itemTagMatrix.numColumns()-1) ? 0 : itemTagMatrix.columnSize(t) ;			
			reg_It_vec.set(t, count > 0 ? 1.0 / Math.sqrt(count) : 1.0);
		}*/		
		
		// To get the embedding for the tags		
		String filename = cf.getString("tag.dict"); //tag-embedding dictionary
		tagtoembedding = loadEmbeddings(filename);		
		
		// get the cumulative embedding for the users
		sumUserTag = new DenseMatrix(numUsers, 300);
		for (int u = 0; u < numUsers ; u++) {
			List<Integer> user_tags = userTagCache.get(u);
			DenseVector sumtag = new DenseVector(300);
			if (user_tags.size() > 0) {
				for (int t : user_tags) {
					sumtag =  sumtag.add(tagtoembedding.get(userTagDao.getItemId(t)).scale(userTagMatrix.get(u,t))); //to give weight according to its frequency 
				}
			}
			for (int i = 0 ; i < 300 ; i++ ) {
				sumUserTag.add(u, i, sumtag.get(i));
			}
		}
		
		// get the cumulative embedding for the items
		sumItemTag = new DenseMatrix(numItems, 300);
		for (int j = 0; j < numItems ; j++) {
			List<Integer> item_tags = itemTagCache.get(j);
			DenseVector sumtag = new DenseVector(300);
			if (item_tags.size() > 0) {
				for (int t : item_tags) {
					sumtag = sumtag.add(tagtoembedding.get(itemTagDao.getItemId(t)).scale(itemTagMatrix.get(j,t))); //here itemid denotes the tagid 
				}
			}
			for (int i = 0 ; i < 300 ; i++ ) {
				sumItemTag.add(j, i, sumtag.get(i));
			}
		}
	}

	@Override
	protected void buildModel() throws Exception {		
		
		alpha = regOptions.getDouble("-alpha", reg);
		beta  = regOptions.getDouble("-beta", reg); 
		regM = regOptions.getDouble("-m", reg);
		mu = regOptions.getDouble("-mu", reg);		
		
		
		loc_lrate = lRate;
		DenseMatrix delta_E = new DenseMatrix(numFactors,300);
		DenseMatrix delta_F = new DenseMatrix(numFactors,300);
		for (int iter = 1; iter <= numIters; iter++) {
			loss = 0; prediter = iter;
			//lRate = loc_lrate;

			for (MatrixEntry me : trainMatrix) {
				int u = me.row(); // user
				int j = me.column(); // item
				double ruj = me.get(); // rating

				double bu = userBias.get(u);
				double bj = itemBias.get(j);				
				
				double reg_Tu = reg_Tu_vec.get(u);
				double reg_Tj = reg_Tj_vec.get(j);
				
				double pred = predict(u,j);
				double euj = pred - ruj;					
				euj =  pred-ruj;
				
				loss += euj * euj;		
				
				// update factors
				double reg_Iu = reg_Iu_vec.get(u);
				double reg_Uj = reg_Uj_vec.get(j);

				double sgd = euj + regB * reg_Iu * bu;
				userBias.add(u, -lRate * sgd);

				sgd = euj + regB * reg_Uj * bj;
				itemBias.add(j, -lRate * sgd);

				loss += regB * reg_Iu * bu * bu;
				loss += regB * reg_Uj * bj * bj;
				
				List<Integer> rated_items = userItemsCache.get(u);
				double[] sum_ys = new double[numFactors]; 
				for (int f = 0; f < numFactors; f++) {
					double sum = 0;
					for (int i : rated_items)
						sum += Y.get(i, f);

					sum_ys[f] = sum * reg_Iu;
				}

				double[] sumEt_arr = sumEt.getData(); 
				double[] sumFt_arr = sumFt.getData(); 
				//System.out.println(sumEt.toString());
				//System.out.println(sumFt.toString());
				
				for (int f = 0; f < numFactors; f++) {
					double puf = P.get(u, f);
					double qjf = Q.get(j, f);
					double wuf = W.get(u, f);
					double xjf = X.get(j, f);

					double delta_u = euj * qjf + regU * reg_Iu * puf;
					double delta_j = euj * (puf + sum_ys[f]) + regI * reg_Uj * qjf;
					double delta_w = euj * alpha * reg_Tu * sumEt_arr[f] + regU * reg_Iu * wuf; //little subtle in the normalization part
					double delta_x = euj * beta  * reg_Tj * sumFt_arr[f] + regI * reg_Uj * xjf;
					
					P.add(u, f, -lRate * delta_u);
					Q.add(j, f, -lRate * delta_j);
					W.add(u, f, -lRate * delta_w);
					X.add(j, f, -lRate * delta_x);

					loss += regU * reg_Iu * puf * puf + regI * reg_Uj * qjf * qjf + regU * reg_Iu * wuf * wuf + regI * reg_Uj * xjf * xjf;

					for (int i : rated_items) {
						double yif = Y.get(i, f);

						double reg_Ui = reg_Uj_vec.get(i);
						double delta_y = euj * qjf * reg_Iu + regI * reg_Ui * yif;
						Y.add(i, f, -lRate * delta_y);

						loss += regI * reg_Ui * yif * yif;
					}
				}
				
				if((iter == 1) || (iter%10==0)) {
				delta_E = W.row(u).outer(sumUserTag.row(u)).scale(alpha * euj * reg_Tu);
				delta_E = delta_E.add(E.scale(regM));
				E = E.add(delta_E.scale(-lRate)); //##
				
				//System.out.println(numUsers + " " + numItems + "  " + u + "  " + j);
				delta_F = X.row(j).outer(sumItemTag.row(j)).scale(beta * euj * reg_Tj);
				delta_F = delta_F.add(F.scale(regM));
				F = F.add(delta_F.scale(-lRate));
				}
			}		
			
			
			loss += regM * E.fro_Norm();
			loss += regM * F.fro_Norm();
			loss *= 0.5;
			

			if (isConverged(iter))
				break;
			
			if (iter>=0 ) //&& iter%5==0
				minmet = evalRatMetrics(bw, "  Dim : " + Float.toString(numFactors) + "  regU : " + Float.toString(regU) + "  regI : " + Float.toString(regI) + "  regB : " + Float.toString(regB) + "  alpha : " + Double.toString(alpha) + "  beta : " + Double.toString(beta) + "  mu : " + Double.toString(mu) + "  m : " + Double.toString(regM) + "  filename : " + cf.getString("trainpath") +"  ",minmet, iter==numIters); //mine
		}// end of training
	}

	@Override
	public double predict(int u, int j) throws Exception {
		//System.out.println("num users : " + numUsers + " train users : " + trainMatrix.numRows());
		//System.out.println("num items : " + numItems + " train users : " + trainMatrix.numColumns());
		//sumEt = 
		if ((u>trainMatrix.numRows()-1) || j>trainMatrix.numColumns()-1)
			return globalMean;
		
		//sumEt
		List<Integer> user_tags = userTagCache.get(u);		
		double reg_Tu = reg_Tu_vec.get(u);				
		if (user_tags.size() > 0) {
			sumEt  = E.mult(sumUserTag.row(u)).scale(reg_Tu * alpha);
		} else {
			//sumEt = 
		}
		
		//sumFt
		List<Integer> item_tags = itemTagCache.get(j); 
		double reg_Tj = reg_Tj_vec.get(j);
		if (item_tags.size() > 0) {
			sumFt  = F.mult(sumItemTag.row(j)).scale(reg_Tj * beta);
		}		
		
		double pred = globalMean + userBias.get(u) + itemBias.get(j) + DenseMatrix.rowMult(P, u, Q, j) + alpha * reg_Tu_vec.get(u) * W.row(u).inner(sumEt) + beta * reg_Tj_vec.get(j) * X.row(j).inner(sumFt);

		// Y
		List<Integer> rated_items = userItemsCache.get(u);		
		if (rated_items.size() > 0) {
			double sum = 0;
			for (int i : rated_items)
				sum += DenseMatrix.rowMult(Y, i, Q, j);

			pred += sum * reg_Iu_vec.get(u);
		}		
		//if (prediter>50) 
			//System.out.println("gb : " + globalMean +" bu : " + userBias.get(u) + " bj : " + itemBias.get(j) + " PutQj : " + (pred - (globalMean + userBias.get(u) + itemBias.get(j))) + " pred : " + pred );
		return pred;
		
	}
	
	private Hashtable<String, DenseVector> loadEmbeddings(String filename) throws IOException {
		
		BufferedReader br = new BufferedReader(new FileReader(filename));
		Hashtable<String, DenseVector> tagtoEmbedding = new Hashtable<String, DenseVector>();
		
		String line;
		while((line = br.readLine()) != null) {
			String[] words = line.split(" ");
			double[] arr = new double[300];
			for (int i=0;i<300;i++) {
				arr[i] = Double.parseDouble(words[i+1]);
			}
			DenseVector embed = new DenseVector(arr);
			tagtoEmbedding.put(words[0], embed);
		}
		br.close();
		return tagtoEmbedding;		
	}	
}

/*
///////////////////////////////////////////////////
if (regOptions.getDouble("-fold", reg) ==1 ) {
	if(Integer.parseInt(rateDao.getUserId(u)) >= 2000) {
		loss += mu * euj * euj;
		euj = mu * euj;
	} 	else {					
		loss += euj * euj;
	}					
} else {
	if(Integer.parseInt(rateDao.getUserId(u)) < 2000) {
		loss += mu * euj * euj;
		euj = mu * euj;
	} 	else {					
		loss += euj * euj;
	}					
}

///////////////////////////////////////////////////////
*/