package org.ea.javacnn.losslayers;

import org.ea.javacnn.data.BackPropResult;
import org.ea.javacnn.data.DataBlock;
import org.ea.javacnn.data.OutputDefinition;
import org.ea.javacnn.layers.Layer;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * Created by danielp on 1/25/17.
 */
public abstract class LossLayer implements Layer,Serializable {
    protected int num_inputs, out_depth, out_sx, out_sy;
    protected DataBlock in_act, out_act;

    public LossLayer(OutputDefinition def) {
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
    public void backward() {}
    public abstract double backward(int y);

    @Override
    public List<BackPropResult> getBackPropagationResult() {
        return new ArrayList<BackPropResult>();
    }

    public DataBlock getOutAct() {
        return this.out_act;
    }
}
