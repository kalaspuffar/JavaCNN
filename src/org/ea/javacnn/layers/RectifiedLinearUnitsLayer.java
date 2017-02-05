package org.ea.javacnn.layers;

import org.ea.javacnn.data.BackPropResult;
import org.ea.javacnn.data.DataBlock;
import org.ea.javacnn.data.OutputDefinition;

import java.util.ArrayList;
import java.util.List;

/**
 * This is a layer of neurons that applies the non-saturating activation
 * function f(x)=max(0,x). It increases the nonlinear properties of the
 * decision function and of the overall network without affecting the
 * receptive fields of the convolution layer.
 *
 * @author Daniel Persson (mailto.woden@gmail.com)
 */
public class RectifiedLinearUnitsLayer implements Layer{
    private DataBlock in_act, out_act;

    @Override
    public DataBlock forward(DataBlock db, boolean training) {
        this.in_act = db;
        DataBlock V2 = db.clone();
        int N = db.getWeights().length;
        double[] V2w = V2.getWeights();
        for(int i=0; i<N; i++) {
            if(V2w[i] < 0) V2.setGradient(i, 0); // threshold at 0
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
            if(V2.getWeight(i) <= 0) {
                V.setGradient(i, 0); // threshold
            } else {
                V.setGradient(i, V2.getGradient(i));
            }
        }
    }

    @Override
    public List<BackPropResult> getBackPropagationResult() {
        return new ArrayList<>();
    }
}