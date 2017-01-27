package org.ea.javacnn;

import org.ea.javacnn.data.BackPropResult;
import org.ea.javacnn.data.DataBlock;
import org.ea.javacnn.layers.Layer;
import org.ea.javacnn.layers.LossLayer;

import java.util.ArrayList;
import java.util.List;

/**
 * A network class holding the layers and some helper functions
 * for training and validation.
 *
 * @author Daniel Persson (mailto.woden@gmail.com)
 */
public class JavaCNN {
    private List<Layer> layers;

    public JavaCNN(List<Layer> layers) {
        this.layers = layers;
    }

    /*
     forward prop the network.
     The trainer class passes is_training = true, but when this function is
     called from outside (not from the trainer), it defaults to prediction mode
    */
    public DataBlock forward(DataBlock db, boolean training) {
      DataBlock act = this.layers.get(0).forward(db, training);
      for(int i=1;i<this.layers.size();i++) {
        act = this.layers.get(i).forward(act, training);
      }
      return act;
    }

    public double getCostLoss(DataBlock db, int y) {
      this.forward(db, false);
      int N = this.layers.size();
      double loss = ((LossLayer)this.layers.get(N-1)).backward(y);
      return loss;
    }

    // backprop: compute gradients wrt all parameters
    public double backward(int y) {
      int N = this.layers.size();
      double loss = ((LossLayer)this.layers.get(N-1)).backward(y);
      for(int i=N-2;i>=0;i--) { // first layer assumed input
        this.layers.get(i).backward();
      }
      return loss;
    }

    // accumulate parameters and gradients for the entire network
    public List<BackPropResult> getBackPropagationResult() {
      List<BackPropResult> result = new ArrayList<BackPropResult>();
      for(Layer l : this.layers) {
        List<BackPropResult> subResult = l.getBackPropagationResult();
        result.addAll(subResult);
      }
      return result;
    }

    /*
    this is a convenience function for returning the argmax
    prediction, assuming the last layer of the net is a softmax
    */
    public int getPrediction() {
      LossLayer S = (LossLayer)this.layers.get(this.layers.size()-1);
      double[] p = S.getOutAct().getWeights();
      double maxv = p[0];
      int maxi = 0;
      for(int i=1; i<p.length; i++) {
        if(p[i] > maxv) {
          maxv = p[i];
          maxi = i;
        }
      }
      return maxi;
    }
}
