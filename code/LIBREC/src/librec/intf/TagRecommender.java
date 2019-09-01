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

import java.io.IOException;
import java.util.List;

import librec.data.Configuration;
import librec.data.DataDAO;
import librec.data.SparseMatrix;
import librec.data.SparseVector;
import librec.util.Logs;
import librec.util.Strings;

import com.google.common.cache.LoadingCache;

/**
 * Recommenders in which social information is used
 * 
 * @author guoguibing
 * 
 */
@Configuration("factors, lRate, maxLRate, regB, regU, regI, regS, iters, boldDriver")
public abstract class TagRecommender extends IterativeRecommender {
	
	
	protected double oldval = Float.MAX_VALUE; 
	// social data dao
	protected static DataDAO userTagDao;
	protected static DataDAO itemTagDao;
	protected static boolean isTag ;
	// socialMatrix: social rate matrix, indicating a user is connecting to a number of other users
	// trSocialMatrix: inverse social matrix, indicating a user is connected by a number of other users
	
	//protected static SparseMatrix socialMatrix;
	protected static SparseMatrix userTagMatrix;
	protected static SparseMatrix itemTagMatrix;

	// a list of rating scales
	protected static List<Double> ratingScaleUT,ratingScaleVT;
	// Maximum, minimum values of rating scales
	protected static double maxRateUT, minRateUT, maxRateVT, minRateVT;
		
	// social regularization
	//protected static float regS;
	protected static float regSU,regSV;
	protected static int numTags;

	// indicator of static field initialization or reset
	public static boolean resetStatics = true;

	// shared social cache for all social recommenders
	protected LoadingCache<Integer, SparseVector> socialCache;
	protected LoadingCache<Integer, List<Integer>> userFriendsCache;
	
	protected LoadingCache<Integer, List<Integer>> userTagCache;
	protected LoadingCache<Integer, List<Integer>> itemTagCache;

	// initialization
	static {
		String tagPath = cf.getPath("dataset.simu");
		String tagPath2 = cf.getPath("dataset.simv");
		Logs.debug("Tag dataset: {}", Strings.last(tagPath, 38));
		Logs.debug("Tag2 dataset: {}", Strings.last(tagPath2, 38));

		if(cf.getString("tagtype").contains("sim")) {
			isTag = false;
			userTagDao = new DataDAO(tagPath, rateDao.getUserIds());
			itemTagDao = new DataDAO(tagPath2, rateDao.getItemIds());
			numUsers = userTagDao.numUsers();
			numItems = itemTagDao.numUsers(); //users are responsible for updated itemids
		} else {
			isTag = true;
			userTagDao = new DataDAO(tagPath, rateDao.getUserIds(), null);
			itemTagDao = new DataDAO(tagPath2, rateDao.getItemIds(), userTagDao.getItemIds()); // here itemids are responsible for tagids mine
			numUsers = userTagDao.numUsers();
			numItems = itemTagDao.numUsers();
		}			

		try {
			userTagMatrix = userTagDao.readData()[0];
			itemTagMatrix = itemTagDao.readData()[0];
			//numUsers  = tagDao.numUsers();
			if(isTag) {
				numTags  = itemTagDao.numItems();
			}

			//socialCache = socialMatrix.rowCache(cacheSpec);
		} catch (Exception e) {
			e.printStackTrace();
			System.exit(-1);
		}
	}

	public TagRecommender(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
		super(trainMatrix, testMatrix, fold);

		if (resetStatics) {
			resetStatics = false;
			regSU = regOptions.getFloat("-su", reg);
			regSV = regOptions.getFloat("-sv", reg);
		}
	}

	public TagRecommender(SparseMatrix trainMatrix, SparseMatrix testMatrix, SparseMatrix validMatrix, int fold) throws IOException {
		super(trainMatrix, testMatrix, validMatrix, fold);
		//mine

		ratingScaleUT = userTagDao.getRatingScale();
		minRateUT = ratingScaleUT.get(0);
		maxRateUT = ratingScaleUT.get(ratingScaleUT.size() - 1);
		ratingScaleVT = itemTagDao.getRatingScale();
		minRateVT = ratingScaleVT.get(0);
		maxRateVT = ratingScaleVT.get(ratingScaleVT.size() - 1);
		if (resetStatics) {
			resetStatics = false;
			regSU = regOptions.getFloat("-su", reg);
			regSV = regOptions.getFloat("-sv", reg);
		}
	}
	
	/**
	 * normalize a rating to the region (0, 1)
	 */
	protected double normalizeTag(double rate, boolean isUT) {
		if (isUT)
			return (rate - minRateUT + 1) / (maxRateUT - minRateUT + 1);
		else
			return (rate - minRateVT +1) / (maxRateVT - minRateVT + 1);
	}
	
	@Override
	public String toString() {
		return Strings.toString(new Object[] { numFactors, initLRate, maxLRate, regB, regU, regI, regSU, regSV, numIters,
				isBoldDriver });
	}

	@Override
	protected boolean isTestable(int u, int j) {
		switch (view) {
		case "cold-start":
			return trainMatrix.rowSize(u) < 5 ? true : false;
		case "trust-degree":
			int min_deg = cf.getInt("min.trust.degree");
			int max_deg = cf.getInt("max.trust.degree");
			if (min_deg == -1)
				min_deg = 0;
			if (max_deg == -1)
				max_deg = Integer.MAX_VALUE;

			// size could be indegree + outdegree
			int in_deg = userTagMatrix.columnSize(u);
			int out_deg = userTagMatrix.rowSize(u);
			int deg = in_deg + out_deg;

			boolean cond = (deg >= min_deg) && (deg <= max_deg);

			return cond ? true : false;

		case "all":
		default:
			return true;
		}
	}

}

