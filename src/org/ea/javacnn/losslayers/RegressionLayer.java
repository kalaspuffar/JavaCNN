package org.ea.javacnn.losslayers;

import org.ea.javacnn.data.DataBlock;
import org.ea.javacnn.data.OutputDefinition;

/**
 * Regression layer is used when your output is an area of data.
 * When you don't have a single class that is the correct activation
 * but you try to find a result set near to your training area.
 *
 * @author Daniel Persson (mailto.woden@gmail.com)
 */
public class RegressionLayer extends LossLayer  {
    public RegressionLayer(OutputDefinition def) {
        super(def);
    }

    @Override
    public DataBlock forward(DataBlock db, boolean training) {
        this.in_act = db;
        this.out_act = db; // nothing to do, output raw scores
        return db;
    }

    public double backward(double[] y) {
        // compute and accumulate gradient wrt weights and bias of this layer
        DataBlock x = this.in_act;
        x.clearGradient(); // zero out the gradient of input Vol
        double loss = 0.0;
        for(int i=0;i<this.out_depth;i++) {
            double dy = x.getWeight(i) - y[i];
            x.setGradient(i, dy);
            loss += 0.5*dy*dy;
        }
        return loss;
    }

    @Override
    public double backward(int y) {
        // compute and accumulate gradient wrt weights and bias of this layer
        DataBlock x = this.in_act;
        x.clearGradient(); // zero out the gradient of input Vol
        double loss = 0.0;

        // lets hope that only one number is being regressed
        double dy = x.getWeight(0) - y;
        x.setGradient(0, dy);
        loss += 0.5*dy*dy;

        return loss;
    }
}
