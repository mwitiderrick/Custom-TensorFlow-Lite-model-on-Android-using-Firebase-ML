package com.namespace.fireml;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.util.Log;
import android.view.View;
import android.widget.ImageView;
import android.widget.TextView;

import com.google.firebase.ml.common.modeldownload.FirebaseModelDownloadConditions;
import com.google.firebase.ml.common.modeldownload.FirebaseModelManager;
import com.google.firebase.ml.custom.FirebaseCustomRemoteModel;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.Interpreter;
import org.tensorflow.lite.support.common.FileUtil;
import org.tensorflow.lite.support.common.TensorProcessor;
import org.tensorflow.lite.support.common.ops.NormalizeOp;
import org.tensorflow.lite.support.image.ImageProcessor;
import org.tensorflow.lite.support.image.TensorImage;
import org.tensorflow.lite.support.image.ops.ResizeOp;
import org.tensorflow.lite.support.label.TensorLabel;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.File;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.util.List;
import java.util.Map;

public class MainActivity extends AppCompatActivity {
    ImageView imageView;
    Context context;
    TextView textView;
    String result = "";
    TensorBuffer modelOutput;
    Interpreter interpreter;
    File modelFile;
    Interpreter.Options options = new Interpreter.Options();

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        imageView = findViewById(R.id.imageView);
        textView = findViewById(R.id.textView);
    }

    public void predict(View view) {
        context = this;
        FirebaseCustomRemoteModel remoteModel = new FirebaseCustomRemoteModel.Builder("MobileNet").build();

        FirebaseModelDownloadConditions conditions = new FirebaseModelDownloadConditions.Builder()
                .requireWifi()
                .build();

        FirebaseModelManager.getInstance().download(remoteModel, conditions)
                .addOnSuccessListener(v -> {
                    Log.i("Info", "Switching to downloaded model");
                    FirebaseModelManager.getInstance().getLatestModelFile(remoteModel)
                            .addOnCompleteListener(task -> {
                                modelFile = task.getResult();
                                assert modelFile != null;
                                interpreter = new Interpreter(modelFile, options);
                            });
                });

        if (modelFile != null) {
            interpreter = new Interpreter(modelFile, options);
            makePrediction();
        } else {
            Log.i("Info", "Trying Local Model");

            try {
                MappedByteBuffer tfliteModel = FileUtil.loadMappedFile(context, "mobilenet_v1_1.0_224_quant.tflite");
                Interpreter.Options options = new Interpreter.Options();
                interpreter = new Interpreter(tfliteModel, options);
                makePrediction();
            } catch (IOException e) {
                Log.e("tflite Support", "Error reading model", e);
            }

        }
    }

    void makePrediction() {
        Bitmap bitmap = BitmapFactory.decodeResource(getResources(), R.drawable.hen);
        ImageProcessor imageProcessor = new ImageProcessor.Builder()
                .add(new ResizeOp(224, 224, ResizeOp.ResizeMethod.BILINEAR))
                .build();
        TensorImage tImage = new TensorImage(DataType.UINT8);
        tImage = imageProcessor.process(TensorImage.fromBitmap(bitmap));
        modelOutput = TensorBuffer.createFixedSize(new int[]{1, 1001}, DataType.UINT8);
        interpreter.run(tImage.getBuffer(), modelOutput.getBuffer());
        final String MOBILE_NET_LABELS = "labels_mobilenet_quant_v1_224.txt";
        List<String> mobilenetlabels = null;
        try {
            mobilenetlabels = FileUtil.loadLabels(context, MOBILE_NET_LABELS);
        } catch (IOException e) {
            Log.e("tfliteSupport", "Error reading label file", e);
        }
        TensorProcessor probabilityProcessor = new TensorProcessor.Builder().add(new NormalizeOp(0, 255)).build();
        if (mobilenetlabels != null) {
            // Map of labels and their corresponding probability
            TensorLabel labels = new TensorLabel(mobilenetlabels, probabilityProcessor.process(modelOutput));
            // Create a map to access the result based on label
            Map<String, Float> resultsMap = labels.getMapWithFloatValue();

            for (String key : resultsMap.keySet()) {
                Float value = resultsMap.get(key);
                if (value >= 0.50) {
                    String roundOff = String.format("%.2f", value);
                    result = key + " " + roundOff;
                }
                Log.i("Info", key + " " + value);
            }
            Log.i("Info", "The label is " + result);
            textView.append(result);

            modelOutput = TensorBuffer.createFixedSize(new int[]{1, 1001}, DataType.UINT8);
        }
    }
}