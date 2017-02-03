package org.ea.javacnn.layers;

import org.ea.javacnn.data.BackPropResult;
import org.ea.javacnn.data.DataBlock;
import org.ea.javacnn.data.OutputDefinition;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * Implements Maxout nonlinearity that computes x -> max(x)
 * where x is a vector of size group_size. Ideally of course,
 * the input size should be exactly divisible by group_size
 *
 * @author Daniel Persson (mailto.woden@gmail.com)
 */
public class MaxoutLayer implements Layer {
    private int out_depth, out_sx, out_sy;

    private DataBlock in_act, out_act;

    private final int group_size = 2;
    private int[] switches;

    public MaxoutLayer(OutputDefinition def) {
        // computed
        this.out_sx = def.getOutX();
        this.out_sy = def.getOutY();
        this.out_depth = (int)Math.floor(def.getDepth() / this.group_size);

        this.switches = new int[this.out_sx*this.out_sy*this.out_depth]; // useful for backprop
        Arrays.fill(this.switches, 0);
    }

    @Override
    public DataBlock forward(DataBlock db, boolean training) {
        this.in_act = db;
        int N = this.out_depth;
        DataBlock V2 = new DataBlock(this.out_sx, this.out_sy, this.out_depth, 0.0);

        // optimization branch. If we're operating on 1D arrays we dont have
        // to worry about keeping track of x,y,d coordinates inside
        // input volumes. In convnets we do :(
        if(this.out_sx == 1 && this.out_sy == 1) {
            for(int i=0;i<N;i++) {
                int ix = i * this.group_size; // base index offset
                double a = db.getWeight(ix);
                int ai = 0;
                for(int j=1;j<this.group_size;j++) {
                    double a2 = db.getWeight(ix+j);
                    if(a2 > a) {
                        a = a2;
                        ai = j;
                    }
                }
                V2.setWeight(i, a);
                this.switches[i] = ix + ai;
            }
        } else {
            int n=0; // counter for switches
            for(int x=0; x< db.getSX(); x++) {
                for(int y=0; y<db.getSY(); y++) {
                    for(int i=0; i<N; i++) {
                        int ix = i * this.group_size;
                        double a = db.getWeight(x, y, ix);
                        int ai = 0;
                        for(int j=1; j<this.group_size; j++) {
                            double a2 = db.getWeight(x, y, ix+j);
                            if(a2 > a) {
                                a = a2;
                                ai = j;
                            }
                        }
                        V2.setWeight(x,y,i,a);
                        this.switches[n] = ix + ai;
                        n++;
                    }
                }
            }

        }
        this.out_act = V2;
        return this.out_act;
    }

    @Override
    public void backward() {
        DataBlock V = this.in_act; // we need to set dw of this
        DataBlock V2 = this.out_act;
        int N = this.out_depth;
        V.clearGradient(); // zero out gradient wrt data

        // pass the gradient through the appropriate switch
        if(this.out_sx == 1 && this.out_sy == 1) {
            for(int i=0; i<N; i++) {
                double chain_grad = V2.getGradient(i);
                V.setGradient(this.switches[i], chain_grad);
            }
        } else {
            // bleh okay, lets do this the hard way
            int n=0; // counter for switches
            for(int x=0; x<V2.getSX(); x++) {
                for(int y=0; y<V2.getSY(); y++) {
                    for(int i=0; i<N; i++) {
                        double chain_grad = V2.getGradient(x,y,i);
                        V.setGradient(x,y,this.switches[n],chain_grad);
                        n++;
                    }
                }
            }
        }
    }

    @Override
    public List<BackPropResult> getBackPropagationResult() {
        return new ArrayList<>();
    }
}
