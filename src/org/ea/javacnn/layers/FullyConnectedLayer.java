package org.ea.javacnn.layers;

import org.ea.javacnn.data.BackPropResult;
import org.ea.javacnn.data.DataBlock;
import org.ea.javacnn.data.OutputDefinition;

import java.util.ArrayList;
import java.util.List;

/**
 * Neurons in a fully connected layer have full connections to all
 * activations in the previous layer, as seen in regular Neural Networks.
 * Their activations can hence be computed with a matrix multiplication
 * followed by a bias offset.
 *
 * @author Daniel Persson (mailto.woden@gmail.com)
 */
public class FullyConnectedLayer implements Layer {
    private double l1_decay_mul = 0.0;
    private double l2_decay_mul = 1.0;

    private DataBlock in_act;
    private DataBlock out_act;

    private final float BIAS_PREF = 0.1f;

    private int out_depth, out_sx, out_sy;
    private int num_inputs;
    private List<DataBlock> filters;
    private DataBlock biases;


    public FullyConnectedLayer(OutputDefinition def, int num_neurons) {
        this.out_depth = num_neurons;

        // computed
        this.num_inputs = def.getOutX() * def.getOutY() * def.getDepth();
        this.out_sx = 1;
        this.out_sy = 1;

        // initializations
        float bias = BIAS_PREF;
        for(int i=0;i<this.out_depth ;i++) {
            this.filters.add(new DataBlock(1, 1, this.num_inputs));
        }
        this.biases = new DataBlock(1, 1, this.out_depth, bias);

        def.setOutX(out_sx);
        def.setOutY(out_sy);
        def.setDepth(out_depth);
    }

    @Override
    public DataBlock forward(DataBlock db, boolean training) {
        this.in_act = db;
        DataBlock A = new DataBlock(1, 1, this.out_depth, 0.0);
        double[] Vw = db.getWeights();
        for(int i=0;i<this.out_depth;i++) {
            double a = 0.0;
            double[] wi = this.filters.get(i).getWeights();
            for(int d=0;d<this.num_inputs;d++) {
                a += Vw[d] * wi[d]; // for efficiency use Vols directly for now
            }
            a += this.biases.getWeight(i);
            A.setWeight(i, a);
        }
        this.out_act = A;
        return this.out_act;
    }

    @Override
    public void backward() {
        DataBlock V = this.in_act;
        V.clearGradient();

        // compute gradient wrt weights and data
        for(int i=0;i<this.out_depth;i++) {
            DataBlock tfi = this.filters.get(i);
            double chain_grad = this.out_act.getGradients()[i];
            for(int d=0;d<this.num_inputs;d++) {
                V.addGradient(d, tfi.getWeight(d)*chain_grad); // grad wrt input data
                tfi.addGradient(d, V.getWeight(d)*chain_grad); // grad wrt params
            }
            this.biases.addGradient(i, chain_grad);
        }
    }

    @Override
    public List<BackPropResult> getBackPropagationResult() {
        List<BackPropResult> results = new ArrayList<BackPropResult>();
        for(int i=0;i<this.out_depth;i++) {
            results.add(new BackPropResult(this.filters.get(i).getWeights(), this.filters.get(i).getGradients(), this.l1_decay_mul, this.l2_decay_mul));
        }
        results.add(new BackPropResult(this.biases.getWeights(), this.biases.getGradients(),  0.0, 0.0));

        return results;
    }
}
