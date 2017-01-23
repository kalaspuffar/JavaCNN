package org.ea.javacnn.layers;

import org.ea.javacnn.data.BackPropResult;
import org.ea.javacnn.data.DataBlock;
import org.ea.javacnn.data.OutputDefinition;

import java.util.ArrayList;
import java.util.List;

/**
 * This layer will squash the result of the activations in the fully
 * connected layer and give you a value of 0 to 1 for all output activations.
 *
 * @author Daniel Persson (mailto.woden@gmail.com)
 */
public class SoftMaxLayer implements Layer {

    private int num_classes;

    public SoftMaxLayer(OutputDefinition def, int num_classes) {
        this.num_classes = num_classes;
    }

    @Override
    public DataBlock forward(DataBlock db, boolean training) {
        return null;
    }

    @Override
    public void backward() {

    }

    @Override
    public List<BackPropResult> getBackPropagationResult() {
        return new ArrayList<BackPropResult>();
    }
}
