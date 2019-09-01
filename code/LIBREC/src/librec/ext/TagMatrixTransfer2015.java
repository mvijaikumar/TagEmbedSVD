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

package librec.ext;

import java.io.IOException;

import librec.data.Configuration;
import librec.data.DenseMatrix;
import librec.data.DenseVector;
import librec.data.MatrixEntry;
import librec.data.SparseMatrix;
import librec.data.SparseVector;
import librec.intf.TagRecommender;
import librec.metric.MetricMine;
import librec.util.Strings;

/**
 * Bo Li, Juncen Li, Jianxin Liao Zhou Fang, Sheng Gao, <strong>Cross-Domain Recommendation via Tag Matrix Transfer </strong>, ICDMW 2015.
 * 
 * @author Vijaikumar M
 * 
 */
@Configuration("factors, numIters")
public class TagMatrixTransfer2015 extends TagRecommender {

	// V = W * H
	protected DenseMatrix W, H;
	protected SparseMatrix V;
	protected SparseMatrix T;
	MetricMine minmet ;

	public TagMatrixTransfer2015(SparseMatrix trainMatrix, SparseMatrix testMatrix, int fold) {
		super(trainMatrix, testMatrix, fold);

		// no need to update learning rate
		lRate = -1;
	}
	public TagMatrixTransfer2015(SparseMatrix trainMatrix, SparseMatrix testMatrix, SparseMatrix validMatrix, int fold) throws IOException {
		super(trainMatrix, testMatrix, validMatrix, fold);

		// no need to update learning rate
		lRate = -1;
	}

	@Override
	protected void initModel() throws Exception {
		W = new DenseMatrix(numUsers, numFactors);
		H = new DenseMatrix(numFactors, numItems);

		W.init(0.01);
		H.init(0.01);

		V = trainMatrix;
	}

	@Override
	protected void buildModel() throws Exception {
		for (int iter = 1; iter <= numIters; iter++) {

			// update W by fixing H
			for (int u = 0; u < W.numRows(); u++) {
				SparseVector uv = V.row(u);

				if (uv.getCount() > 0) {
					SparseVector euv = new SparseVector(V.numColumns());

					for (int j : uv.getIndex())
						euv.set(j, predict(u, j));

					for (int f = 0; f < W.numColumns(); f++) {
						DenseVector fv = H.row(f, false);
						double real = fv.inner(uv);
						double estm = fv.inner(euv) + 1e-9;

						W.set(u, f, W.get(u, f) * (real / estm));
					}
				}
			}

			// update H by fixing W
			DenseMatrix trW = W.transpose();
			for (int j = 0; j < H.numColumns(); j++) {
				SparseVector jv = V.column(j);

				if (jv.getCount() > 0) {
					SparseVector ejv = new SparseVector(V.numRows());

					for (int u : jv.getIndex())
						ejv.set(u, predict(u, j));

					for (int f = 0; f < H.numRows(); f++) {
						DenseVector fv = trW.row(f, false);
						double real = fv.inner(jv);
						double estm = fv.inner(ejv) + 1e-9;

						H.set(f, j, H.get(f, j) * (real / estm));
					}
				}
			}

			// compute errors
			loss = 0;
			for (MatrixEntry me : V) {
				int u = me.row();
				int j = me.column();
				double ruj = me.get();

				if (ruj > 0) {
					double euj = predict(u, j) - ruj;

					loss += euj * euj;
				}
			}

			loss *= 0.5;

			if (isConverged(iter))
				break;
			minmet = evalRatMetrics(bw, "  regU:" + Float.toString(regU) +"  filename : "+" " ,minmet, iter==numIters); //mine 
		}
		System.out.println("asdf regU : "+regU +"  regI : "+ regI+ "  regSU : "  + algoName);
		bw.close();
	}

	@Override
	public double predict(int u, int j) {
		return DenseMatrix.product(W, u, H, j);
	}

	@Override
	public String toString() {
		return Strings.toString(new Object[] { numFactors, numIters });
	}
}
