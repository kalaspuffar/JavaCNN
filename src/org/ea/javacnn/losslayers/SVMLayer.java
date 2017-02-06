package org.ea.javacnn.losslayers;

import org.ea.javacnn.data.DataBlock;
import org.ea.javacnn.data.OutputDefinition;

/**
 * This layer uses the input area trying to find a line to
 * separate the correct activation from the incorrect ones.
 *
 * @author Daniel Persson (mailto.woden@gmail.com)
 */
public class SVMLayer extends LossLayer {

    public SVMLayer(OutputDefinition def) {
        super(def);
    }

    @Override
    public DataBlock forward(DataBlock db, boolean training) {
        this.in_act = db;
        this.out_act = db; // nothing to do, output raw scores
        return db;
    }

    @Override
    public double backward(int y) {
        // compute and accumulate gradient wrt weights and bias of this layer
        DataBlock x = this.in_act;
        x.clearGradient();

        // we're using structured loss here, which means that the score
        // of the ground truth should be higher than the score of any other
        // class, by a margin
        double yscore = x.getWeight(y); // score of ground truth
        double margin = 1.0;
        double loss = 0.0;
        for(int i=0;i<this.out_depth;i++) {
            if(y == i) { continue; }
            double ydiff = -yscore + x.getWeight(i) + margin;
            if(ydiff > 0) {
                // violating dimension, apply loss
                x.addGradient(i,1);
                x.subGradient(y,1);
                loss += ydiff;
            }
        }
        return loss;
    }
}
