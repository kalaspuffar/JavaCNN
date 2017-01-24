package org.ea.javacnn.layers;

import org.ea.javacnn.data.BackPropResult;
import org.ea.javacnn.data.DataBlock;
import org.ea.javacnn.data.OutputDefinition;

import java.util.ArrayList;
import java.util.List;

/**
 * This layer uses different filters to find attributes of the data that
 * affects the result. As an example there could be a filter to find
 * horizontal edges in an image.
 *
 * @author Daniel Persson (mailto.woden@gmail.com)
 */
public class ConvolutionLayer implements Layer {
    private final float l1_decay_mul = 0.0f;
    private final float l2_decay_mul = 1.0f;

    private DataBlock in_act;
    private DataBlock out_act;

    private final float BIAS_PREF = 0.1f;

    private int out_depth, out_sx, out_sy;
    private int in_depth, in_sx, in_sy;
    private int sx, sy;
    private int stride, padding;
    private List<DataBlock> filters;
    private DataBlock biases;

    public ConvolutionLayer(OutputDefinition def, int sx, int filters, int stride, int padding) {
        // required
        this.out_depth = filters;
        this.sx = sx; // filter size. Should be odd if possible, it's cleaner.
        this.in_depth = def.getDepth();
        this.in_sx = def.getOutX();
        this.in_sy = def.getOutY();

        // optional
        this.sy = this.sx;
        this.stride = stride;
        this.padding = padding;

        // computed
        // note we are doing floor, so if the strided convolution of the filter doesnt fit into the input
        // volume exactly, the output volume will be trimmed and not contain the (incomplete) computed
        // final application.
        this.out_sx = (int)Math.floor((this.in_sx + this.padding * 2 - this.sx) / this.stride + 1);
        this.out_sy = (int)Math.floor((this.in_sy + this.padding * 2 - this.sy) / this.stride + 1);

        // initializations
        this.filters = new ArrayList<DataBlock>();
        for(int i=0;i<this.out_depth;i++) {
            this.filters.add(new DataBlock(this.sx, this.sy, this.in_depth));
        }
        this.biases = new DataBlock(1, 1, this.out_depth, BIAS_PREF);

    }

    @Override
    public DataBlock forward(DataBlock db, boolean training) {
        this.in_act = db;
        DataBlock A = new DataBlock(this.out_sx, this.out_sy, this.out_depth, 0.0);

        int V_sx = this.in_sx;
        int V_sy = this.in_sy;
        int xy_stride = this.stride;

        for(int d=0;d<this.out_depth;d++) {
            DataBlock f = this.filters.get(d);
            int y = -this.padding;
            for(int ay=0; ay<this.out_sy; y+=xy_stride,ay++) {  // xy_stride
                int x = -this.padding;
                for(int ax=0; ax<this.out_sx; x+=xy_stride,ax++) {  // xy_stride

                    // convolve centered at this particular location
                    double a = 0.0;
                    for(int fy=0;fy<f.getSY();fy++) {
                        int oy = y+fy; // coordinates in the original input array coordinates
                        for(int fx=0;fx<f.getSX();fx++) {
                            int ox = x+fx;
                            if(oy>=0 && oy<V_sy && ox>=0 && ox<V_sx) {
                                for(int fd=0;fd<f.getDepth();fd++) {
                                    // avoid function call overhead (x2) for efficiency, compromise modularity :(
                                    a += f.getWeight(fx, fy, fd) * db.getWeight(ox, oy, fd);
                                }
                            }
                        }
                    }
                    a += this.biases.getWeight(d);
                    A.setWeight(ax, ay, d, a);
                }
            }
        }
        this.out_act = A;
        return A;
    }

    @Override
    public void backward() {
        DataBlock db = this.in_act;
        db.clearGradient(); // zero out gradient wrt bottom data, we're about to fill it
        int V_sx = db.getSX();
        int V_sy = db.getSY();
        int xy_stride = this.stride;

        for(int d=0;d<this.out_depth;d++) {
            DataBlock f = this.filters.get(d);
            int y = -this.padding;
            for(var ay=0; ay<this.out_sy; y+=xy_stride,ay++) {  // xy_stride
                int x = -this.padding;
                for(var ax=0; ax<this.out_sx; x+=xy_stride,ax++) {  // xy_stride

                    // convolve centered at this particular location
                    float chain_grad = this.out_act.getGradient(ax,ay,d); // gradient from above, from chain rule
                    for(int fy=0;fy<f.getSY();fy++) {
                        int oy = y+fy; // coordinates in the original input array coordinates
                        for(int fx=0;fx<f.getSX();fx++) {
                            int ox = x+fx;
                            if(oy>=0 && oy<V_sy && ox>=0 && ox<V_sx) {
                                for(var fd=0;fd<f.getDepth();fd++) {
                                    // avoid function call overhead (x2) for efficiency, compromise modularity :(
                                    var ix1 = ((V_sx * oy)+ox)*V.getDepth()+fd;
                                    var ix2 = ((f.getSY() * fy)+fx)*f.getDepth()+fd;
                                    f.addGradient(ix2, V.getWeight(ix1)*chain_grad);
                                    V.addGradient(ix1, f.getWeight(ix2)*chain_grad);
                                }
                            }
                        }
                    }
                    this.biases.addGradient(d, chain_grad);
                }
            }
        }
    }

    @Override
    public List<BackPropResult> getBackPropagationResult() {

        /*
        var response = [];
      for(var i=0;i<this.out_depth;i++) {
        response.push({params: this.filters[i].w, grads: this.filters[i].dw, l2_decay_mul: this.l2_decay_mul, l1_decay_mul: this.l1_decay_mul});
      }
      response.push({params: this.biases.w, grads: this.biases.dw, l1_decay_mul: 0.0, l2_decay_mul: 0.0});
      return response;
         */

        return new ArrayList<BackPropResult>();
    }
}
