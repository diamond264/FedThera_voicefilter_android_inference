package com.example.voicefilter_inference_2

import android.content.Context
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.os.Environment
import android.util.Log
import flanagan.math.FourierTransform
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder

class MainActivity : AppCompatActivity() {
    val TAG: String = "VOICEFILTER_log"

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        Log.d(TAG, "onCreate")

        val dvec_model_file = File(this.getExternalFilesDir(null), "embedder_test3.pt")
        val vf_model_file: File = File(this.getExternalFilesDir(null), "vf_inference_test3.pt")
        val dvec_module = Module.load(dvec_model_file.toString())
        val vf_module = Module.load(vf_model_file.toString())

        // Reading audio from .wav file
        var audioIS = FileInputStream(File(this.getExternalFilesDir(null), "test.wav"))
        val audioBytes = ByteArrayOutputStream()
        while (audioIS.available() > 0) {
            audioBytes.write(audioIS.read())
        }
        val tmpBytes = audioBytes.toByteArray()
        var audioL16Samples = ShortArray(tmpBytes.size / 2)
        ByteBuffer.wrap(tmpBytes).order(ByteOrder.LITTLE_ENDIAN).asShortBuffer().get(audioL16Samples)
        audioIS.close()

        Log.d(TAG, "samples:")
        audioL16Samples = audioL16Samples.slice(22..(audioL16Samples.size-1)).toShortArray() // for wav

        // Processing mono audio
//        val monoSamples = ShortArray(audioL16Samples.size / 2)
//        for (i in monoSamples.indices) {
//            monoSamples[i] =
//                    (((audioL16Samples.get(i * 2) + audioL16Samples.get(i * 2 + 1)) / 2).toShort())
//        }
//        audioL16Samples = monoSamples
//        Log.d(TAG, audioL16Samples[0].toString())

        // Now it has the same result with python librosa.load
        var audio_raw = audioL16Samples.map{(it/32768.0).toDouble()}.toDoubleArray()

        Log.d(TAG, "?")


        val wlen = 4000
        val wshift = 2000

//        var beg_samp = 0
//        var end_samp = wlen // WLEN
//        var N_fr = (audio_raw.size - wlen)/wshift
//        sig_arr = audio_raw.slice(beg_samp..(end_samp-1)).toFloatArray()
//        beg_samp = beg_samp + wshift
//        end_samp = beg_samp + wlen
//        count_fr_tot = count_fr_tot + 1
//        var inpTensor = Tensor.fromBlob(sig_arr, sig_arr_shape)
//        var inp = IValue.from(inpTensor)
//        var outTensor = module.forward(inp).toTensor().dataAsFloatArray
////            Log.d(TAG, outTensor.getDataAsFloatArray()[0].toString())
//        var maxScore = -java.lang.Float.MAX_VALUE
//        var maxScoreIdx = -1
//        for (i in 0 until outTensor.size) {
//            if (outTensor[i] > maxScore) {
//                maxScore = outTensor[i]
//                maxScoreIdx = i
//            }
//        }
//        Log.d(TAG, maxScoreIdx.toString())
//        maxIdxList.add(maxScoreIdx)
    }
}