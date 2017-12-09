package org.ea.javacnn.data;

import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.Arrays;
import java.util.Random;

/**
 * Holding all the data handled by the network. So a layer will receive
 * this class and return a similar block as a output that will be used
 * by the next layer in the chain.
 *
 * @author Daniel Persson (mailto.woden@gmail.com)
 */
public class DataBlock {
    private int sx;
    private int sy;
    private int depth;
    private double[] w;
    private double[] dw;

    public DataBlock(int sx, int sy, int depth) {
        this(sx, sy, depth, -1.0);
    }

    public DataBlock(int sx, int sy, int depth, double c) {
        this.sx = sx;
        this.sy = sy;
        this.depth = depth;
        int n = sx * sy * depth;

        this.w = new double[n];
        this.dw = new double[n];
        if(c != -1.0) {
            Arrays.fill(this.w, c);
        } else {
            double scale = Math.sqrt(1.0 / n);
            Random r = new Random();
            for(int i=0; i<n; i++) {
                this.w[i] = r.nextDouble() * scale;
            }
        }
        Arrays.fill(this.dw, 0);
    }

    public void addImageData(byte[] imgData, int maxvalue) {
        // prepare the input: get pixels and normalize them
        for(int i=0;i<imgData.length;i++) {
            w[i] = ((imgData[i] & 0xff)/(float)maxvalue)-0.5; // normalize image pixels to [-0.5, 0.5]
        }
    }

    public int getSX() {
        return sx;
    }

    public int getSY() {
        return sy;
    }

    public int getDepth() {
        return depth;
    }

    public double getWeight(int ix) {
        return this.w[ix];
    }

    public double getWeight(int x, int y, int depth) {
        int ix = ((this.sx * y) + x) * this.depth + depth;
        return this.w[ix];
    }

    public void setWeight(int ix, double val) {
        this.w[ix] = val;
    }

    public void setWeight(int x, int y, int depth, double val) {
        int ix = ((this.sx * y) + x) * this.depth + depth;
        setWeight(ix, val);
    }

    public void addWeight(int x, int y, int depth, double val) {
        int ix = ((this.sx * y) + x) * this.depth + depth;
        this.w[ix] += val;
    }

    public double getGradient(int x, int y, int depth) {
        int ix = ((this.sx * y) + x) * this.depth + depth;
        return getGradient(ix);
    }

    public double getGradient(int ix) {
        return this.dw[ix];
    }

    public void setGradient(int x, int y, int depth, double val) {
        int ix = ((this.sx * y) + x) * this.depth + depth;
        setGradient(ix, val);
    }

    public void setGradient(int ix, double val) {
        this.dw[ix] = val;
    }


    public void addGradient(int x, int y, int depth, double val) {
        int ix = ((this.sx * y) + x) * this.depth + depth;
        this.addGradient(ix, val);
    }

    public void addGradient(int ix, double val) {
        this.dw[ix] += val;
    }

    public void subGradient(int ix, double val) {
        this.dw[ix] -= val;
    }

    public void mulGradient(int ix, double val) {
        this.dw[ix] *= val;
    }


    public DataBlock cloneAndZero() {
        return new DataBlock(this.sx, this.sy, this.depth, 0.0);
    }

    public DataBlock clone() {
        DataBlock db = new DataBlock(this.sx, this.sy, this.depth, 0.0);
        for(int i=0; i<this.w.length; i++) {
            db.w[i] = this.w[i];
        }
        return db;
    }

    public void clearGradient() {
        Arrays.fill(this.dw, 0);
    }

    public double[] getWeights() {
        return w;
    }

    public double[] getGradients() {
        return dw;
    }

    public void addFrom(DataBlock db) {
        for(int i=0; i<this.w.length; i++) {
            this.w[i] = db.w[i];
        }
    }

    public void addFromScaled(DataBlock db, double a) {
        for(int i=0; i<this.w.length; i++) {
            this.w[i] = db.w[i] * a;
        }
    }

    public void setConst(double a) {
        for(int i=0; i<this.w.length; i++) {
            this.w[i] = a;
        }
    }
}
