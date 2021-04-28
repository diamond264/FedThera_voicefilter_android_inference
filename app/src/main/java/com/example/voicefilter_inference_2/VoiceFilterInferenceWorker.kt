package com.example.voicefilter_inference_2

import android.app.NotificationManager
import android.content.Context
import android.os.Build
import android.util.Log
import androidx.annotation.RequiresApi
import androidx.core.app.NotificationCompat
import androidx.work.CoroutineWorker
import androidx.work.ForegroundInfo
import androidx.work.WorkManager
import androidx.work.WorkerParameters
import org.jtransforms.fft.DoubleFFT_1D
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import java.io.ByteArrayOutputStream
import java.io.File
import java.io.FileInputStream
import java.io.FileNotFoundException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.*
import kotlin.math.log10
import kotlin.math.pow
import kotlin.math.sqrt

class VoiceFilterInferenceWorker(context: Context, parameters: WorkerParameters) :
        CoroutineWorker(context, parameters) {

//    private val notificationManager =
//            context.getSystemService(Context.NOTIFICATION_SERVICE) as
//                    NotificationManager
    val TAG: String = "VOICEFILTER_log"
    val ctx = context

    fun parse_mel_file(path: String): Array<DoubleArray> {
        val file = File(path)
        val csv_array = ArrayList<DoubleArray>()
        val inputStream: Scanner
        try {
            inputStream = Scanner(file)
            while (inputStream.hasNext()) {
                val line = inputStream.next()
                val values = line.split(",").toTypedArray()
                val sub_array = ArrayList<Double>()
                for (value in values) {
                    sub_array.add(value.toDouble())
                }
                csv_array.add(sub_array.toDoubleArray())
            }
            inputStream.close()
        } catch (e: FileNotFoundException) {
            e.printStackTrace()
        }
        return csv_array.toTypedArray()
    }

    fun STFT(sig: DoubleArray, n_fft: Int, hop_length: Int, win_length: Int): Pair<Array<DoubleArray>, Array<DoubleArray>> {
        // Creating 'hanning' window
        var window = DoubleArray(n_fft) { 0.0 }
        for (i in 0..win_length-1) window[i+(n_fft-win_length)/2] =
                0.5 - 0.5 * Math.cos(2 * Math.PI * i / (win_length))

        var padded_sig = DoubleArray(sig.size + n_fft) { 0.0 }
        for (i in padded_sig.indices) {
            if (i < n_fft/2) padded_sig[i] = sig[n_fft/2-i]
            else if (i < padded_sig.size-n_fft/2) padded_sig[i] = sig[i-n_fft/2]
            else padded_sig[i] = sig[sig.size-(i-padded_sig.size+n_fft/2)-2]
        }

        val sec = 1+((padded_sig.size-n_fft + 0.0)/hop_length).toInt()
        var framed_audio = Array(sec) { DoubleArray(n_fft) { 0.0 } }
        for (i in 0..sec-1) {
            for (j in 0..n_fft-1) {
                framed_audio[i][j] = (padded_sig[j + hop_length*i]) * window[j]
            }
        }

        var stft_result_r = Array(sec) { DoubleArray(n_fft/2+1) { 0.0 } }
        var stft_result_i = Array(sec) { DoubleArray(n_fft/2+1) { 0.0 } }
        for (i in 0 until sec) {
            val a = DoubleArray(2 * framed_audio[i].size) //fft 수행할 배열 사이즈 2N
            for (k in 0 until framed_audio[i].size) {
                a[2 * k] = framed_audio[i][k] //Re
                a[2 * k + 1] = 0.0 //Im
            }

            val fft_new = DoubleFFT_1D(n_fft.toLong())
            fft_new.complexForward(a)

            for (k in 0 until framed_audio[i].size/2+1) {
                stft_result_r[i][k] = a[2 * k] //Re
                stft_result_i[i][k] = a[2 * k + 1] //Im
            }
        }

        return Pair(stft_result_r, stft_result_i)
    }

    fun get_mel(sig: DoubleArray, sr: Int, n_mels: Int, n_fft: Int, hop_length: Int, win_length: Int): FloatArray {
        val stft_result = STFT(sig, n_fft, hop_length, win_length)
        val stft_r = stft_result.first
        val stft_i = stft_result.second
        var magnitudes = Array(stft_r.size) { DoubleArray(stft_r[0].size) { 0.0 } }
        for (i in 0 until stft_r.size) { // sec
            for (j in 0 until stft_r[0].size) {
                magnitudes[i][j] = stft_r[i][j].pow(2) + stft_i[i][j].pow(2)
            }
        }

        val mel_basis_file = File(ctx.getExternalFilesDir(null), "mel_basis.csv")
        val mel_basis = parse_mel_file(mel_basis_file.toString()) // 40 x 257

        var mel = FloatArray(magnitudes.size * mel_basis.size) { 0.0F }
        for (i in 0 until magnitudes.size) { // sec
            for (j in 0 until mel_basis.size) { // n_mels (40)
                var sum = 0.0
                for (k in 0 until magnitudes[i].size) { // n_fft/2 + 1
                    sum += magnitudes[i][k] * mel_basis[j][k]
                }
                mel[j*magnitudes.size+i] = log10(sum+1e-6).toFloat()
            }
        }

        return mel
    }

    fun amp_to_db(stft_r: Array<DoubleArray>, stft_i: Array<DoubleArray>): Array<DoubleArray> {
        val ref_level_db = 20.0
        var stft_abs = Array(stft_r.size) { DoubleArray(stft_r[0].size) { 0.0 } }
        for (i in 0 until stft_abs.size) {
            for (j in 0 until stft_abs[i].size) {
                stft_abs[i][j] = sqrt(stft_r[i][j].pow(2) + stft_i[i][j].pow(2))
                if (stft_abs[i][j] < 1e-5) stft_abs[i][j] = 1e-5
                stft_abs[i][j] = log10(stft_abs[i][j]) * 20 - ref_level_db
            }
        }

        return stft_abs
    }

    fun angle(stft_r: Array<DoubleArray>, stft_i: Array<DoubleArray>): Array<DoubleArray> {
        var stft_angle = Array(stft_r.size) { DoubleArray(stft_r[0].size) { 0.0 } }
        for (i in 0 until stft_angle.size) {
            for (j in 0 until stft_angle[i].size) {
                stft_angle[i][j] = Math.atan2(stft_i[i][j], stft_r[i][j])
            }
        }

        return stft_angle
    }

    fun normalize(sig: Array<DoubleArray>): Array<DoubleArray> {
        val min_level_db = -100.0
        var stft_normalize = Array(sig.size) { DoubleArray(sig[0].size) { 0.0 } }
        for (i in 0 until stft_normalize.size) {
            for (j in 0 until stft_normalize[i].size) {
                stft_normalize[i][j] = sig[i][j]/(-min_level_db)
                if (stft_normalize[i][j] < -1.0) stft_normalize[i][j] = -1.0
                if (stft_normalize[i][j] > 0.0) stft_normalize[i][j] = 0.0
                stft_normalize[i][j] = stft_normalize[i][j] + 1.0
            }
        }

        return stft_normalize
    }

    fun wav2spec(sig: DoubleArray, n_fft: Int, hop_length: Int, win_length: Int): Pair<Array<DoubleArray>, Array<DoubleArray>> {
        val stft_result = STFT(sig, n_fft, hop_length, win_length)
        val stft_r = stft_result.first
        val stft_i = stft_result.second

        var S = amp_to_db(stft_r, stft_i)
        S = normalize(S)
        var D = angle(stft_r, stft_i)

        return Pair(S, D)
    }

    override suspend fun doWork(): Result {
        inference()
        return Result.success()
    }

    private fun inference() {
        val dvec_model_file = File(ctx.getExternalFilesDir(null), "embedder_test3.pt")
        val vf_model_file: File = File(ctx.getExternalFilesDir(null), "vf_lighten_test3.pt")
        val dvec_module = Module.load(dvec_model_file.toString())
        val vf_module = Module.load(vf_model_file.toString())

        // Reading audio from .wav file
        var audioIS = FileInputStream(File(ctx.getExternalFilesDir(null), "test.wav"))
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

        val dvec_mel = get_mel(audio_raw, 16000, 40, 512, 160, 400)
        val dvec_shape = arrayOf<Long>(40, (dvec_mel.size/40).toLong()).toLongArray()

        val start_time = System.currentTimeMillis()
        val dvec_tensor = Tensor.fromBlob(dvec_mel, dvec_shape)
        val dvec_inp = IValue.from(dvec_tensor)
        val dvec_out = dvec_module.forward(dvec_inp).toTensor().dataAsFloatArray
        Log.d(TAG, (System.currentTimeMillis()-start_time).toString())

        val spec = wav2spec(audio_raw, 1200, 160, 400)
        val mag = spec.first
        val phase = spec.second

        var mag_flatten = FloatArray(mag.size*mag[0].size) {0.0F}
        for (i in 0 until mag.size) {
            for (j in 0 until mag[i].size) mag_flatten[i*mag[i].size+j] = mag[i][j].toFloat()
        }
        val mag_shape = arrayOf<Long>(1, (mag_flatten.size/601).toLong(), 601).toLongArray()
        val mag_tensor = Tensor.fromBlob(mag_flatten, mag_shape)
        val mag_inp = IValue.from(mag_tensor)

        val vf_dvec_shape = arrayOf<Long>(1, 256).toLongArray()
        val vf_dvec_tensor = Tensor.fromBlob(dvec_out, vf_dvec_shape)
        val vf_dvec_inp = IValue.from(vf_dvec_tensor)
        val vf_out = vf_module.forward(mag_inp, vf_dvec_inp)
        Log.d(TAG, vf_out.toStr())
    }

    @RequiresApi(Build.VERSION_CODES.O)
    private fun createChannel() {
        // Create a Notification channel
    }
}