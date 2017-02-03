package org.ea.javacnn.layers;

import org.ea.javacnn.data.BackPropResult;
import org.ea.javacnn.data.DataBlock;
import org.ea.javacnn.data.OutputDefinition;

import java.util.ArrayList;
import java.util.List;

/**
 * Implements Tanh nonlinearity elementwise x -> tanh(x)
 * so the output is between -1 and 1.
 *
 * @author Daniel Persson (mailto.woden@gmail.com)
 */
public class TanhLayer implements Layer {
    private DataBlock in_act, out_act;

    @Override
    public DataBlock forward(DataBlock db, boolean training) {
        this.in_act = db;
        DataBlock V2 = db.cloneAndZero();
        int N = db.getWeights().length;
        for(int i=0; i<N; i++) {
            V2.setWeight(i, Math.tanh(db.getWeight(i)));
        }
        this.out_act = V2;
        return this.out_act;
    }

    @Override
    public void backward() {
        DataBlock V = this.in_act; // we need to set dw of this
        DataBlock V2 = this.out_act;
        int N = V.getWeights().length;
        V.clearGradient(); // zero out gradient wrt data
        for(int i=0; i<N; i++) {
            double v2wi = V2.getWeight(i);
            V.setGradient(i,(1.0 - v2wi * v2wi) * V2.getGradient(i));
        }
    }

    @Override
    public List<BackPropResult> getBackPropagationResult() {
        return new ArrayList<>();
    }
}
