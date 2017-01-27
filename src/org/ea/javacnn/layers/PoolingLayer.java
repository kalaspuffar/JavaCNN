package org.ea.javacnn.layers;

import org.ea.javacnn.data.BackPropResult;
import org.ea.javacnn.data.DataBlock;
import org.ea.javacnn.data.OutputDefinition;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * This layer will reduce the dataset by creating a smaller zoomed out
 * version. In essence you take a cluster of pixels take the sum of them
 * and put the result in the reduced position of the new image.
 *
 * @author Daniel Persson (mailto.woden@gmail.com)
 */
public class PoolingLayer implements Layer {
    private int in_depth, in_sx, in_sy;
    private int out_depth, out_sx, out_sy;
    private int sx, sy, stride, padding;

    private int[] switchx;
    private int[] switchy;

    private DataBlock in_act, out_act;

    public PoolingLayer(OutputDefinition def, int sx, int stride, int padding) {
        this.sx = sx;
        this.stride = stride;

        this.in_depth = def.getDepth();
        this.in_sx = def.getOutX();
        this.in_sy = def.getOutY();

        // optional
        this.sy = this.sx;
        this.padding = padding;

        // computed
        this.out_depth = this.in_depth;
        this.out_sx = (int)Math.floor((this.in_sx + this.padding * 2 - this.sx) / this.stride + 1);
        this.out_sy = (int)Math.floor((this.in_sy + this.padding * 2 - this.sy) / this.stride + 1);

        // store switches for x,y coordinates for where the max comes from, for each output neuron
        this.switchx = new int[this.out_sx*this.out_sy*this.out_depth];
        this.switchy = new int[this.out_sx*this.out_sy*this.out_depth];
        Arrays.fill(switchx, 0);
        Arrays.fill(switchy, 0);

        def.setOutX(out_sx);
        def.setOutY(out_sy);
        def.setDepth(out_depth);
    }

    @Override
    public DataBlock forward(DataBlock db, boolean training) {
        this.in_act = db;

        DataBlock A = new DataBlock(this.out_sx, this.out_sy, this.out_depth, 0.0);

        int n=0; // a counter for switches
        for(int d=0;d<this.out_depth;d++) {
            int x = -this.padding;
            for(int ax=0; ax<this.out_sx; x+=this.stride,ax++) {
                int y = -this.padding;
                for(int ay=0; ay<this.out_sy; y+=this.stride,ay++) {

                    // convolve centered at this particular location
                    double a = -99999; // hopefully small enough ;\
                    int winx=-1;
                    int winy=-1;
                    for(int fx=0;fx<this.sx;fx++) {
                        for(int fy=0;fy<this.sy;fy++) {
                            int oy = y+fy;
                            int ox = x+fx;
                            if(oy>=0 && oy<db.getSY() && ox>=0 && ox<db.getSX()) {
                                double v = db.getWeight(ox, oy, d);
                                // perform max pooling and store pointers to where
                                // the max came from. This will speed up backprop
                                // and can help make nice visualizations in future
                                if(v > a) {
                                    a = v;
                                    winx=ox;
                                    winy=oy;
                                }
                            }
                        }
                    }
                    this.switchx[n] = winx;
                    this.switchy[n] = winy;
                    n++;
                    A.setWeight(ax, ay, d, a);
                }
            }
        }
        this.out_act = A;
        return this.out_act;
    }

    @Override
    public void backward() {
        // pooling layers have no parameters, so simply compute
        // gradient wrt data here
        DataBlock V = this.in_act;
        V.clearGradient(); // zero out gradient wrt data

        int n = 0;
        for(int d=0;d<this.out_depth;d++) {
            int x = -this.padding;
            for(int ax=0; ax<this.out_sx; x+=this.stride,ax++) {
                int y = -this.padding;
                for(int ay=0; ay<this.out_sy; y+=this.stride,ay++) {

                    double chain_grad = this.out_act.getGradient(ax,ay,d);
                    V.addGradient(this.switchx[n], this.switchy[n], d, chain_grad);
                    n++;

                }
            }
        }
    }

    @Override
    public List<BackPropResult> getBackPropagationResult() {
        return new ArrayList<BackPropResult>();
    }
}
