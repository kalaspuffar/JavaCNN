package org.ea.javacnn.layers;

import org.ea.javacnn.data.BackPropResult;
import org.ea.javacnn.data.DataBlock;
import org.ea.javacnn.data.OutputDefinition;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * This layer will remove some random activations in order to
 * defeat over-fitting.
 *
 * @author Daniel Persson (mailto.woden@gmail.com)
 */
public class DropoutLayer implements Layer,Serializable {
    private int out_depth, out_sx, out_sy;

    private DataBlock in_act, out_act;

    private final double drop_prob = 0.5;
    private boolean[] dropped;

    public DropoutLayer(OutputDefinition def) {
        // computed
        this.out_sx = def.getOutX();
        this.out_sy = def.getOutY();
        this.out_depth = def.getDepth();

        this.dropped = new boolean[this.out_sx*this.out_sy*this.out_depth];
    }

    @Override
    public DataBlock forward(DataBlock db, boolean training) {
        this.in_act = db;
        DataBlock V2 = db.clone();
        int N = db.getWeights().length;
        if(training) {
            // do dropout
            for(int i=0;i<N;i++) {
                if(Math.random()<this.drop_prob) {
                    V2.setWeight(i, 0);
                    this.dropped[i] = true;
                } else {
                    // drop!
                    this.dropped[i] = false;
                }
            }
        } else {
            // scale the activations during prediction
            for(int i=0;i<N;i++) {
                V2.mulGradient(i, this.drop_prob);
            }
        }
        this.out_act = V2;
        return this.out_act; // dummy identity function for now
    }

    @Override
    public void backward() {
        DataBlock V = this.in_act; // we need to set dw of this
        DataBlock chain_grad = this.out_act;
        int N = V.getWeights().length;
        V.clearGradient(); // zero out gradient wrt data
        for(int i=0;i<N;i++) {
            if(!(this.dropped[i])) {
                V.setGradient(i, chain_grad.getGradient(i)); // copy over the gradient
            }
        }
    }

    @Override
    public List<BackPropResult> getBackPropagationResult() {
        return new ArrayList<>();
    }
}
