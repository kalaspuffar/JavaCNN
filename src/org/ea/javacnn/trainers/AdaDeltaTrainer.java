package org.ea.javacnn.trainers;

import org.ea.javacnn.JavaCNN;
import org.ea.javacnn.data.BackPropResult;

import java.util.Arrays;

/**
 * Adaptive delta will look at the differences between the expected result and the current result to train the network.
 *
 * @author Daniel Persson (mailto.woden@gmail.com)
 */
public class AdaDeltaTrainer extends Trainer {
    private final double ro = 0.95;

    public AdaDeltaTrainer(JavaCNN net, int batch_size, float l2_decay) {
        super(net, batch_size, l2_decay);
    }

    @Override
    public void initTrainData(BackPropResult bpr) {
        double[] newXSumArr = new double[bpr.getWeights().length];
        Arrays.fill(newXSumArr, 0);
        this.xsum.add(newXSumArr);
    }


    @Override
    public void update(int i, int j, double gij, double[] p) {
        double[] gsumi = this.gsum.get(i);
        double[] xsumi = this.xsum.get(i);
        gsumi[j] = this.ro * gsumi[j] + (1-this.ro) * gij * gij;
        double dx = - Math.sqrt((xsumi[j] + this.eps)/(gsumi[j] + this.eps)) * gij;
        xsumi[j] = this.ro * xsumi[j] + (1-this.ro) * dx * dx; // yes, xsum lags behind gsum by 1.
        p[j] += dx;
    }
}

