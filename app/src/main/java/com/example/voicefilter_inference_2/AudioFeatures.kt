//package com.example.voicefilter_inference_2
//
//import com.jlibrosa.audio.JLibrosa
//import org.jtransforms.fft.DoubleFFT_1D
//import kotlin.math.pow
//
//public class AudioFeatures(audio_path: String, sample_rate: Int, duration: Int) {
//    public val audio = JLibrosa().loadAndRead(audio_path, sample_rate, duration)
//    private val sample_rate = sample_rate
//
//    fun getMagStatistics(): FloatArray {
//        val mean = audio.average().toFloat()
//        var min = Float.POSITIVE_INFINITY
//        var max = Float.NEGATIVE_INFINITY
//        var std = 0.0F
//        for (i in 0 until audio.size) {
//            std += (audio[i]-mean).pow(2)
//            if (audio[i] > max) max = audio[i]
//            if (audio[i] < min) min = audio[i]
//        }
//
//        return arrayOf(mean, min, max, std).toFloatArray()
//    }
//
//    fun getMFCCFeatures(nMFCC: Int): Array<FloatArray> {
//        val mfcc_features = JLibrosa().generateMFCCFeatures(audio, sample_rate, nMFCC)
//        return mfcc_features
//    }
//
////    fun getSTFTFeatures(nMFCC: Int): Array<out Array<org.apache.commons.math3.complex.Complex>> {
////        val stft_features = JLibrosa().generateSTFTFeatures(audio, sample_rate, nMFCC)
////        return stft_features
////    }
//
//    fun getmelspectrogramFeatures(): Array<DoubleArray> {
//        val mel_features = JLibrosa().generateMelSpectroGram(audio)
//        return mel_features
//    }
//}