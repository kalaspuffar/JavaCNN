package org.ea.javacnn.trainers;

import org.ea.javacnn.data.DataBlock;
import org.ea.javacnn.data.TrainResult;

/**
 * Trainers take the generated output of activations and gradients in
 * order to modify the weights in the network to make a better prediction
 * the next time the network runs with a data block.
 *
 * @author Daniel Persson (mailto.woden@gmail.com)
 */
public interface Trainer {
    TrainResult train(DataBlock x, int y);
}
