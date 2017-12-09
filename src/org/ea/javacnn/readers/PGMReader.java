package org.ea.javacnn.readers;

import java.io.*;
import java.util.*;

public class PGMReader implements Reader {
    private int imageSizeX = -1;
    private int imageSizeY = -1;
    private int maxvalue = -1;
    private List<String> labels;
    private String imagePath;
    private List<String> filenames = new ArrayList<>();
    private boolean readLabel = false;
    private boolean readImage = false;
    private int currentImage = 0;

    public PGMReader(String imagePath) {
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
            FileReader fr = new FileReader(new File(imagePath, filename));
            BufferedReader br = new BufferedReader(fr);
            String line;
            boolean foundHeader = false;
            while ((line = br.readLine()) != null) {
                if (line.equals("P5")) {
                    foundHeader = true;
                } else if (line.startsWith("#")) {
                    continue;
                } else if (line.matches("[0-9]+ [0-9]+")) {
                    String[] sizes = line.split(" ");
                    imageSizeX = Integer.parseInt(sizes[0]);
                    imageSizeY = Integer.parseInt(sizes[1]);
                } else if (line.matches("[0-9]+")) {
                    maxvalue = Integer.parseInt(line);
                } else {
                    throw new Exception("Unsupported file format");
                }
                if (maxvalue != -1 && foundHeader && imageSizeX != -1) break;
            }

            br.close();
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
            // Now parse the file as binary data
            FileInputStream fis = new FileInputStream(new File(imagePath, filename));
            DataInputStream dis = new DataInputStream(fis);
            // look for 4 lines (i.e.: the header) and discard them
            int numNewlines = 3;
            while (numNewlines > 0) {
                char c;
                do {
                    c = (char) (dis.readUnsignedByte());
                } while (c != '\n');
                numNewlines--;
            }
            // read the image data
            int[] returnData = new int[imageSizeX * imageSizeY];
            for (int i = 0; i < imageSizeX * imageSizeY; i++) {
                returnData[i] = dis.readUnsignedByte();
            }

            if(readLabel) {
                incrementCounter();
            } else {
                readImage = true;
            }
            return returnData;
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
        Reader mr = new PGMReader("pgmfiles/train");
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
