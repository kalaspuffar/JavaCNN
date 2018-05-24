package org.ea.javacnn.layers;

import org.ea.javacnn.data.BackPropResult;
import org.ea.javacnn.data.DataBlock;
import org.ea.javacnn.data.OutputDefinition;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * This layer is useful when we are dealing with ReLU neurons. Why is that?
 * Because ReLU neurons have unbounded activations and we need LRN to normalize
 * that. We want to detect high frequency features with a large response. If we
 * normalize around the local neighborhood of the excited neuron, it becomes even
 * more sensitive as compared to its neighbors.
 *
 * At the same time, it will dampen the responses that are uniformly large in any
 * given local neighborhood. If all the values are large, then normalizing those
 * values will diminish all of them. So basically we want to encourage some kind
 * of inhibition and boost the neurons with relatively larger activations. This
 * has been discussed nicely in Section 3.3 of the original paper by Krizhevsky et al.
 *
 * @author Daniel Persson (mailto.woden@gmail.com)
 */
public class LocalResponseNormalizationLayer implements Layer,Serializable {

    /*
     The constants k, n, alpha and beta are hyper-parameters whose
       values are determined using a validation set; we used
       k = 2, n = 5, alpha = 10^-4, beta = 0.75

       quote from http://www.cs.toronto.edu/~fritz/absps/imagenet.pdf
    */

    private final double k = 2.0;
    private final double n = 5.0;
    private final double alpha = 0.0001;
    private final double beta = 0.75;

    private DataBlock in_act, out_act, S_cache_;

    public LocalResponseNormalizationLayer() {
        // checks
        if(this.n%2 == 0) {
            System.out.println("WARNING n should be odd for LRN layer");
        }
    }

    @Override
    public DataBlock forward(DataBlock db, boolean training) {
        this.in_act = db;

        DataBlock A = db.cloneAndZero();
        this.S_cache_ = db.cloneAndZero();
        double n2 = Math.floor(this.n/2);
        for(int x=0;x<db.getSX();x++) {
            for(int y=0;y<db.getSY();y++) {
                for(int i=0;i<db.getDepth();i++) {

                    double ai = db.getWeight(x,y,i);

                    // normalize in a window of size n
                    double den = 0.0;
                    for(int j=(int)Math.max(0,i-n2);j<=Math.min(i+n2,db.getDepth()-1);j++) {
                        double aa = db.getWeight(x,y,j);
                        den += aa*aa;
                    }
                    den *= this.alpha / this.n;
                    den += this.k;
                    this.S_cache_.setWeight(x,y,i,den); // will be useful for backprop
                    den = Math.pow(den, this.beta);
                    A.setWeight(x,y,i,ai/den);
                }
            }
        }

        this.out_act = A;
        return this.out_act; // dummy identity function for now
    }

    @Override
    public void backward() {
        // evaluate gradient wrt data
        DataBlock V = this.in_act; // we need to set dw of this
        V.clearGradient();

        int n2 = (int)Math.floor(this.n/2);
        for(int x=0;x<V.getSX();x++) {
            for(int y=0;y<V.getSY();y++) {
                for(int i=0;i<V.getDepth();i++) {

                    double chain_grad = this.out_act.getGradient(x,y,i);
                    double S = this.S_cache_.getWeight(x,y,i);
                    double SB = Math.pow(S, this.beta);
                    double SB2 = SB*SB;

                    // normalize in a window of size n
                    for(int j=(int)Math.max(0,i-n2);j<=Math.min(i+n2,V.getDepth()-1);j++) {
                        double aj = V.getWeight(x,y,j);
                        double g = -aj*this.beta*Math.pow(S,this.beta-1)*this.alpha/this.n*2*aj;
                        if(j==i) g+= SB;
                        g /= SB2;
                        g *= chain_grad;
                        V.addGradient(x,y,j,g);
                    }

                }
            }
        }
    }

    @Override
    public List<BackPropResult> getBackPropagationResult() {
        return new ArrayList<BackPropResult>();
    }
}
