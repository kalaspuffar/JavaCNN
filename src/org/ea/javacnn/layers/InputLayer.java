package org.ea.javacnn.layers;

import org.ea.javacnn.data.BackPropResult;
import org.ea.javacnn.data.DataBlock;

import java.util.ArrayList;
import java.util.List;

/**
 * The input layer is a simple layer that will pass the data though and
 * create a window into the full training data set. So for instance if
 * we have an image of size 28x28x1 which means that we have 28 pixels
 * in the x axle and 28 pixels in the y axle and one color (gray scale),
 * then this layer might give you a window of another size example 24x24x1
 * that is randomly chosen in order to create some distortion into the
 * dataset so the algorithm don't over-fit the training.
 *
 * @author Daniel Persson (mailto.woden@gmail.com)
 */
public class InputLayer implements Layer {

    private int out_sx;
    private int out_sy;
    private int out_depth;

    public InputLayer(int out_sx, int out_sy, int out_depth) {
        this.out_sx = out_sx;
        this.out_sy = out_sy;
        this.out_depth = out_depth;
    }

    @Override
    public void forward(DataBlock db, boolean training) {

    }

    @Override
    public void backward() {

    }

    @Override
    public List<BackPropResult> getBackPropagationResult() {
        return new ArrayList<BackPropResult>();
    }
}
