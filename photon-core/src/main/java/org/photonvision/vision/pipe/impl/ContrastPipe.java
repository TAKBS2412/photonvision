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

import org.opencv.core.Mat;
import org.photonvision.vision.pipe.CVPipe;

public class ContrastPipe extends CVPipe<Mat, Mat, ContrastPipe.ContrastParams> {

    @Override
    protected Mat process(Mat in) {

        double alpha = 2.0;

        Mat newMat = Mat.zeros(in.size(), in.type());
        byte[] imageData = new byte[(int) (in.total() * in.channels())];
        byte[] newImageData = new byte[(int) (in.total() * in.channels())];

        in.get(0, 0, imageData);

        for (int i = 0; i < imageData.length; i++) {
            int convertedImageData = imageData[i];
            if (convertedImageData < 0) {
                convertedImageData += 256;
            }
            // -128 = (128, 128, 128)
            // -127 = (129, 129, 129)
            // -1 = (255, 255, 255)
            // 0 = (0, 0, 0)
            // 127 = (127, 127, 127)
            newImageData[i] = saturate(convertedImageData * alpha);
            // newImageData[i] = (byte) -1;
        }

        newMat.put(0, 0, newImageData);

        return newMat;
    }

    private byte saturate(double val) {
        int intVal = (int) Math.round(val);
        intVal = Math.max(Math.min(intVal, 255), 0);
        return (byte) intVal;
    }

    // TODO: Add contrast value?
    public static class ContrastParams {
        public ContrastParams() {}
    }
}
