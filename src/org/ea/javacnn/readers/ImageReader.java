package org.ea.javacnn.readers;

import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class ImageReader implements Reader {
    private int imageSizeX = -1;
    private int imageSizeY = -1;
    private int maxvalue = -1;
    private List<String> labels;
    private String imagePath;
    private List<String> filenames = new ArrayList<>();
    private boolean readLabel = false;
    private boolean readImage = false;
    private int currentImage = 0;

    public ImageReader(String imagePath) {
        this.imagePath = imagePath;
        File dir = new File(imagePath);
        Set<String> labels = new HashSet<>();
        for(File f : dir.listFiles()) {
            if(!f.isFile()) continue;
            filenames.add(f.getName());
            labels.add(f.getName().split("_")[0]);
        }

        this.labels = new ArrayList<>();
        this.labels.addAll(labels);

        try {
            String filename = filenames.get(0);
            BufferedImage bi = ImageIO.read(new File(imagePath, filename));
            this.imageSizeX = bi.getWidth();
            this.imageSizeY = bi.getHeight();
            this.maxvalue = 255;
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void incrementCounter() {
        currentImage++;
        readLabel = false;
        readImage = false;
    }

    @Override
    public int readNextLabel() {
        if(readLabel) {
            incrementCounter();
        }
        String filename = filenames.get(currentImage);
        int label = labels.indexOf(filename.split("_")[0]);
        if(readImage) {
            incrementCounter();
        } else {
            readLabel = true;
        }
        return label;
    }

    @Override
    public int[] readNextImage() throws Exception {
        if(readImage) {
            incrementCounter();
        }
        if(currentImage >= this.size()) return new int[0];

        String filename = filenames.get(currentImage);
        try {
            BufferedImage orgImg = ImageIO.read(new File(imagePath, filename));
            BufferedImage newImg = new BufferedImage(imageSizeX, imageSizeY, BufferedImage.TYPE_BYTE_INDEXED);
            Graphics g = newImg.getGraphics();
            g.drawImage(orgImg, 0, 0, null);

            int[] imageData = new int[imageSizeY * imageSizeY];
            imageData = newImg.getData().getPixels(0, 0, imageSizeY, imageSizeY, imageData);
            if(readLabel) {
                incrementCounter();
            } else {
                readImage = true;
            }
            return imageData;
        } catch (Exception e) {
            throw e;
        }
    }

    @Override
    public void reset() {
        currentImage = 0;
        readLabel = false;
        readImage = false;
    }

    public int numOfClasses() {
        return labels.size();
    }

    @Override
    public int size() {
        return filenames.size();
    }

    @Override
    public int getSizeX() {
        return imageSizeX;
    }

    @Override
    public int getSizeY() {
        return imageSizeY;
    }

    @Override
    public int getMaxvalue() {
        return maxvalue;
    }

    public static void main(String[] argv) {
        Reader mr = new ImageReader("pngfiles/train");
        System.out.println(mr.size());
        for(int i=0; i<mr.size(); i++) {
            System.out.print(mr.readNextLabel());
        }

        mr.reset();
        for(int i=0; i<mr.size(); i++) {
            try {
                mr.readNextImage();
            } catch (Exception e) {
                e.printStackTrace();
                System.out.println("Crash at "+i);
            }
        }

        mr.reset();
        try {
            int[] b = mr.readNextImage();
            for(int j=0; j<b.length; j++) {
                if(j % mr.getSizeX() == 0) System.out.println();
                System.out.print((b[j] & 0xFF) + " ");
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
