/*
 * Copyright (C) Photon Vision.
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

package org.photonvision.vision.pipe.impl;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import org.opencv.core.Mat;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.photonvision.vision.pipe.CVPipe;

public class ContrastPipe extends CVPipe<Mat, Mat, ContrastPipe.ContrastParams> {

    // Map of gamma to LUT, where gamma is represented as tenths (i.e., 0.9 is represented as the integer 9).
    private final Map<Integer, Mat> tenthsToLookUpTable = new HashMap<>();

    private int doubleToTenths(double d) {
      return (int) Math.round(d * 10);
    }

    @Override
    protected Mat process(Mat in) {

        double alpha = params.contrastMultiplier;
        int alphaAsTenths = doubleToTenths(alpha);
        Mat lookUpTable = tenthsToLookUpTable.get(alphaAsTenths);
        if (lookUpTable == null) {
            lookUpTable = new Mat(1, 256, CvType.CV_8U);
            byte[] lookUpTableData = new byte[(int) (lookUpTable.total() * lookUpTable.channels())];
            for (int i = 0; i < lookUpTable.cols(); i++) {
                lookUpTableData[i] = saturate(Math.pow(i / 255.0, alpha) * 255.0);
            }
            lookUpTable.put(0, 0, lookUpTableData);
            tenthsToLookUpTable.put(alphaAsTenths, lookUpTable);
        }

        Mat newMat = Mat.zeros(in.size(), in.type());
        Core.LUT(in, lookUpTable, newMat);

        return newMat;
    }

    private byte saturate(double val) {
        int intVal = (int) Math.round(val);
        intVal = Math.max(Math.min(intVal, 255), 0);
        return (byte) intVal;
    }

    public static class ContrastParams {
        public static final double DEFAULT_CONTRAST_MULTIPLIER = 2.0;

        public double contrastMultiplier;

        public ContrastParams() {
            this(DEFAULT_CONTRAST_MULTIPLIER);
        }

        public ContrastParams(double contrastMultiplier) {
            this.contrastMultiplier = contrastMultiplier;
        }
    }
}
