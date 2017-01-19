package org.ea.javacnn.layers;

import org.ea.javacnn.data.BackPropResult;
import org.ea.javacnn.data.DataBlock;

import java.util.ArrayList;
import java.util.List;

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
