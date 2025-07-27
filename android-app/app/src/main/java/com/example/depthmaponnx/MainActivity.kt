package com.example.depthmaponnx

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.drawable.BitmapDrawable
import android.os.Bundle
import android.widget.Button
import android.widget.ImageView
import androidx.appcompat.app.AppCompatActivity
import com.microsoft.onnxruntime.*
import java.nio.FloatBuffer

class MainActivity : AppCompatActivity() {

    private lateinit var inputImageView: ImageView
    private lateinit var depthImageView: ImageView
    private lateinit var ortEnv: OrtEnvironment
    private lateinit var session: OrtSession

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        inputImageView = findViewById(R.id.inputImageView)
        depthImageView = findViewById(R.id.depthImageView)

        findViewById<Button>(R.id.buttonSelect).setOnClickListener {
            selectPhoto()
        }

        findViewById<Button>(R.id.buttonProcess).setOnClickListener {
            processImage()
        }

        initOrt()
    }

    private fun selectPhoto() {
        val intent = Intent(Intent.ACTION_GET_CONTENT)
        intent.type = "image/*"
        startActivityForResult(Intent.createChooser(intent, "Select Picture"), 1)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == 1 && resultCode == Activity.RESULT_OK) {
            val uri = data?.data ?: return
            val inputStream = contentResolver.openInputStream(uri)
            val bitmap = BitmapDrawable(resources, inputStream).bitmap
            inputImageView.setImageBitmap(bitmap)
        }
    }

    private fun initOrt() {
        ortEnv = OrtEnvironment.getEnvironment()
        // Load MiDaS small model from assets (see README for download instructions)
        val modelBytes = assets.open("midas_small.onnx").readBytes()
        val opts = SessionOptions()
        session = ortEnv.createSession(modelBytes, opts)
    }

    private fun processImage() {
        val drawable = inputImageView.drawable as? BitmapDrawable ?: return
        val inputBitmap = drawable.bitmap

        val modelInput = preprocess(inputBitmap)
        val inputName = session.inputNames.iterator().next()
        val tensor = OnnxTensor.createTensor(ortEnv, modelInput)

        val results = session.run(mapOf(inputName to tensor))
        val depthTensor = results[0].value as Array<Array<FloatArray>>

        val depthBitmap = postprocess(depthTensor[0])
        depthImageView.setImageBitmap(depthBitmap)
    }

    private fun preprocess(bitmap: Bitmap): Array<Array<Array<FloatArray>>> {
        val targetWidth = 256
        val targetHeight = 256
        val resized = Bitmap.createScaledBitmap(bitmap, targetWidth, targetHeight, true)
        val imgData = Array(1) { Array(3) { Array(targetHeight) { FloatArray(targetWidth) } } }

        for (y in 0 until targetHeight) {
            for (x in 0 until targetWidth) {
                val pixel = resized.getPixel(x, y)
                imgData[0][0][y][x] = (Color.red(pixel) / 255.0f - 0.5f) / 0.5f
                imgData[0][1][y][x] = (Color.green(pixel) / 255.0f - 0.5f) / 0.5f
                imgData[0][2][y][x] = (Color.blue(pixel) / 255.0f - 0.5f) / 0.5f
            }
        }
        return imgData
    }

    private fun postprocess(depth: Array<FloatArray>): Bitmap {
        val width = depth[0].size
        val height = depth.size
        val min = depth.flatten().minOrNull() ?: 0f
        val max = depth.flatten().maxOrNull() ?: 1f

        val bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        for (y in 0 until height) {
            for (x in 0 until width) {
                val v = (depth[y][x] - min) / (max - min)
                val color = colorMap(v)
                bmp.setPixel(x, y, color)
            }
        }
        return bmp
    }

    private fun colorMap(value: Float): Int {
        val r = (value * 255).toInt()
        val g = 0
        val b = ((1 - value) * 255).toInt()
        return Color.rgb(r, g, b)
    }
}
