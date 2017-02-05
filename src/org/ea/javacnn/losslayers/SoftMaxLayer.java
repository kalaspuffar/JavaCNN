package org.ea.javacnn.losslayers;

import org.ea.javacnn.data.BackPropResult;
import org.ea.javacnn.data.DataBlock;
import org.ea.javacnn.data.OutputDefinition;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * This layer will squash the result of the activations in the fully
 * connected layer and give you a value of 0 to 1 for all output activations.
 *
 * @author Daniel Persson (mailto.woden@gmail.com)
 */
public class SoftMaxLayer extends LossLayer {
    private int num_inputs, out_depth, out_sx, out_sy;

    private double[] es;

    public SoftMaxLayer(OutputDefinition def) {
        super(def);
        // computed
        this.num_inputs = def.getOutY() * def.getOutX() * def.getDepth();
        this.out_depth = this.num_inputs;
        this.out_sx = 1;
        this.out_sy = 1;

        def.setOutX(out_sx);
        def.setOutY(out_sy);
        def.setDepth(out_depth);
    }

    @Override
    public DataBlock forward(DataBlock db, boolean training) {
        this.in_act = db;

        DataBlock A = new DataBlock(1, 1, this.out_depth, 0.0);

        // compute max activation
        double[] as = db.getWeights();
        double amax = db.getWeight(0);
        for(int i=1;i<this.out_depth;i++) {
            if(as[i] > amax) amax = as[i];
        }

        // compute exponentials (carefully to not blow up)
        double[] es = new double[this.out_depth];
        Arrays.fill(es, 0);
        double esum = 0.0;
        for(int i=0;i<this.out_depth;i++) {
            double e = Math.exp(as[i] - amax);
            esum += e;
            es[i] = e;
        }

        // normalize and output to sum to one
        for(int i=0;i<this.out_depth;i++) {
            es[i] /= esum;
            A.setWeight(i, es[i]);
        }

        this.es = es; // save these for backprop
        this.out_act = A;
        return this.out_act;
    }

    @Override
    public double backward(int y) {

        // compute and accumulate gradient wrt weights and bias of this layer
        DataBlock x = this.in_act;
        x.clearGradient(); // zero out the gradient of input Vol

        for(int i=0;i<this.out_depth;i++) {
            double indicator = i == y ? 1.0 : 0.0;
            double mul = -(indicator - this.es[i]);
            x.setGradient(i, mul);
        }

        // loss is the class negative log likelihood
        return -Math.log(this.es[y]);
    }
}