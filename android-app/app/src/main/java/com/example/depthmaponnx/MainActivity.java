package com.example.depthmaponnx;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.SeekBar;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import java.io.IOException;

public class MainActivity extends AppCompatActivity {
    private static final int REQUEST_IMAGE = 100;
    private static final int REQUEST_PERMISSION = 200;

    private ImageView inputImageView;
    private ImageView depthImageView;
    private ImageView chromoImageView;
    private Button buttonSelect;
    private Button buttonProcess;
    private Button buttonClear;

    private SeekBar thresholdSeek;
    private SeekBar depthScaleSeek;
    private SeekBar featherSeek;
    private SeekBar redSeek;
    private SeekBar blueSeek;
    private SeekBar gammaSeek;
    private SeekBar blackSeek;
    private SeekBar whiteSeek;
    private SeekBar smoothingSeek;

    private Bitmap originalBitmap;
    private float[][] currentDepth;

    private DepthEstimator depthEstimator;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        inputImageView = findViewById(R.id.inputImageView);
        depthImageView = findViewById(R.id.depthImageView);
        chromoImageView = findViewById(R.id.chromoImageView);
        buttonSelect = findViewById(R.id.buttonSelect);
        buttonProcess = findViewById(R.id.buttonProcess);
        buttonClear = findViewById(R.id.buttonClear);

        thresholdSeek = findViewById(R.id.thresholdSeek);
        depthScaleSeek = findViewById(R.id.depthScaleSeek);
        featherSeek = findViewById(R.id.featherSeek);
        redSeek = findViewById(R.id.redSeek);
        blueSeek = findViewById(R.id.blueSeek);
        gammaSeek = findViewById(R.id.gammaSeek);
        blackSeek = findViewById(R.id.blackSeek);
        whiteSeek = findViewById(R.id.whiteSeek);
        smoothingSeek = findViewById(R.id.smoothingSeek);

        for (SeekBar sb : new SeekBar[]{thresholdSeek, depthScaleSeek, featherSeek, redSeek, blueSeek, gammaSeek, blackSeek, whiteSeek, smoothingSeek}) {
            sb.setMax(100);
            sb.setOnSeekBarChangeListener(new SeekBar.OnSeekBarChangeListener() {
                @Override
                public void onProgressChanged(SeekBar seekBar, int progress, boolean fromUser) {
                    updateEffect();
                }

                @Override
                public void onStartTrackingTouch(SeekBar seekBar) {}

                @Override
                public void onStopTrackingTouch(SeekBar seekBar) {}
            });
        }

        buttonSelect.setOnClickListener(v -> selectImage());
        buttonProcess.setOnClickListener(v -> generateDepthMap());
        buttonClear.setOnClickListener(v -> clearResults());

        try {
            depthEstimator = new DepthEstimator(getAssets());
        } catch (Exception e) {
            Toast.makeText(this, "Depth model not loaded", Toast.LENGTH_LONG).show();
        }
    }

    private void selectImage() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.READ_EXTERNAL_STORAGE) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, REQUEST_PERMISSION);
            return;
        }
        Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
        intent.setType("image/*");
        startActivityForResult(intent, REQUEST_IMAGE);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);
        if (requestCode == REQUEST_PERMISSION && grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            selectImage();
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == REQUEST_IMAGE && resultCode == RESULT_OK && data != null) {
            Uri uri = data.getData();
            try {
                originalBitmap = MediaStore.Images.Media.getBitmap(getContentResolver(), uri);
                inputImageView.setImageBitmap(originalBitmap);
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    private void generateDepthMap() {
        if (originalBitmap == null || depthEstimator == null) return;
        try {
            currentDepth = depthEstimator.estimateDepth(originalBitmap);
            Bitmap depthBitmap = DepthEstimator.toGrayBitmap(currentDepth);
            depthImageView.setImageBitmap(depthBitmap);
            applyDefaultEffect();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    private void applyDefaultEffect() {
        if (originalBitmap == null || currentDepth == null) return;
        Bitmap effect = ChromoStereoizer.applyEffect(
                originalBitmap,
                currentDepth,
                50, 50, 10,
                50, 50,
                50,
                0, 100,
                0
        );
        chromoImageView.setImageBitmap(effect);
    }

    private void updateEffect() {
        if (originalBitmap == null || currentDepth == null) return;
        Bitmap effect = ChromoStereoizer.applyEffect(
                originalBitmap,
                currentDepth,
                thresholdSeek.getProgress(),
                depthScaleSeek.getProgress(),
                featherSeek.getProgress(),
                redSeek.getProgress(),
                blueSeek.getProgress(),
                gammaSeek.getProgress(),
                blackSeek.getProgress(),
                whiteSeek.getProgress(),
                smoothingSeek.getProgress()
        );
        chromoImageView.setImageBitmap(effect);
    }

    private void clearResults() {
        originalBitmap = null;
        currentDepth = null;
        inputImageView.setImageDrawable(null);
        depthImageView.setImageDrawable(null);
        chromoImageView.setImageDrawable(null);
    }
}
