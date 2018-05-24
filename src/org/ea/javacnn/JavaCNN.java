package org.ea.javacnn;

import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import java.io.PrintWriter;
import java.io.Serializable;
import org.ea.javacnn.data.BackPropResult;
import org.ea.javacnn.data.DataBlock;
import org.ea.javacnn.layers.Layer;
import org.ea.javacnn.losslayers.LossLayer;
import java.util.ArrayList;
import java.util.List;

/**
 * A network class holding the layers and some helper functions
 * for training and validation.
 *
 * @author Daniel Persson (mailto.woden@gmail.com) and s.chekanov 
 */
public class JavaCNN implements Serializable {

    private static final long serialVersionUID = 1L;
    private List<Layer> layers;

    public JavaCNN(List<Layer> layers) {
        this.layers = layers;
    }

    /*
     Forward prop the network.
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


    /** Backprop: compute gradients wrt all parameters
    */
    public double backward(int y) {
      int N = this.layers.size();
      double loss = ((LossLayer)this.layers.get(N-1)).backward(y);
      for(int i=N-2;i>=0;i--) { // first layer assumed input
        this.layers.get(i).backward();
      }
      return loss;
    }

    /**
    * Accumulate parameters and gradients for the entire network
    */
    public List<BackPropResult> getBackPropagationResult() {
      List<BackPropResult> result = new ArrayList<BackPropResult>();
      for(Layer l : this.layers) {
        List<BackPropResult> subResult = l.getBackPropagationResult();
        result.addAll(subResult);
      }
      return result;
    }

    /**
    * This is a convenience function for returning the argmax
    * prediction, assuming the last layer of the net is a softmax
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



        /**
         * Save convolutional network to a file.
         * 
         * @param fileName output file name.  
         */
        public void saveModel(String fileName) {
                try {
                        ObjectOutputStream oos = new ObjectOutputStream(
                                        new FileOutputStream(fileName));
                        oos.writeObject(this);
                        oos.flush();
                        oos.close();
                } catch (IOException e) {
                        e.printStackTrace();
                }

        }

  /**
         * Load the model of this neural network from a file. 
         * 
         * @param fileName input file name 
         * @return
         */
        public static JavaCNN loadModel(String fileName) {
                try {
                        ObjectInputStream in = new ObjectInputStream(new FileInputStream(
                                        fileName));
                        JavaCNN cnn = (JavaCNN) in.readObject();
                        in.close();
                        return cnn;
                } catch (IOException | ClassNotFoundException e) {
                        e.printStackTrace();
                }
                return null;
        }


}
