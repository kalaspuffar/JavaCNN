package org.ea.javacnn.trainers;

import org.ea.javacnn.JavaCNN;
import org.ea.javacnn.data.BackPropResult;

import java.util.Arrays;

/**
 * Created by danielp on 2/3/17.
 */
public class AdamTrainer extends Trainer {
    private final double beta1 = 0.9;
    private final double beta2 = 0.999;

    public AdamTrainer(JavaCNN net, int batch_size, float l2_decay) {
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
        gsumi[j] = gsumi[j] * this.beta1 + (1- this.beta1) * gij; // update biased first moment estimate
        xsumi[j] = xsumi[j] * this.beta2 + (1-this.beta2) * gij * gij; // update biased second moment estimate
        double biasCorr1 = gsumi[j] * (1 - Math.pow(this.beta1, this.k)); // correct bias first moment estimate
        double biasCorr2 = xsumi[j] * (1 - Math.pow(this.beta2, this.k)); // correct bias second moment estimate
        double dx =  - this.learning_rate * biasCorr1 / (Math.sqrt(biasCorr2) + this.eps);
        p[j] += dx;
    }
}
