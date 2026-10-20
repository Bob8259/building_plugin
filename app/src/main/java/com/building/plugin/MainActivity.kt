package com.building.plugin

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.building.plugin.ocr.PpOcrEngine
import com.building.plugin.service.DetectorService
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlin.math.roundToInt

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        // Auto-start the detector service when the activity launches
        val serviceIntent = Intent(this, DetectorService::class.java)
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            startForegroundService(serviceIntent)
        } else {
            startService(serviceIntent)
        }

        setContent {
            MaterialTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    ServiceInfo(modifier = Modifier.padding(innerPadding))
                }
            }
        }
    }
}

@Composable
fun ServiceInfo(modifier: Modifier = Modifier) {
    val context = LocalContext.current
    var resultText by remember { mutableStateOf("") }
    var isRunning by remember { mutableStateOf(false) }
    val scope = rememberCoroutineScope()

    fun runOcrTest() {
        if (isRunning) return
        scope.launch {
            isRunning = true
            resultText = "正在预热识别模型与准备图片…"
            try {
                withContext(Dispatchers.Default) {
                    PpOcrEngine.initialize(context)

                    // 1. 预先加载 Recognition 模型和字典（确保加载耗时不计入纯推理时间）
                    PpOcrEngine.loadRec()

                    // 2. 解码测试图片并裁剪指定范围 [49, 139, 117, 162]
                    val options = BitmapFactory.Options().apply {
                        inPreferredConfig = Bitmap.Config.ARGB_8888
                    }
                    val fullBitmap = context.assets.open("example.png").use { stream ->
                        BitmapFactory.decodeStream(stream, null, options)
                    } ?: throw IllegalStateException("无法从 assets 中加载 example.png")

                    val left = 49
                    val top = 139
                    val right = 117
                    val bottom = 162
                    val width = right - left
                    val height = bottom - top

                    val cropBitmap = try {
                        Bitmap.createBitmap(fullBitmap, left, top, width, height)
                    } finally {
                        fullBitmap.recycle()
                    }

                    // 3. 仅对 Recognition 模型推理进行计时（不计入加载和裁剪时间）
                    val result = try {
                        val start = System.currentTimeMillis()
                        val recResult = PpOcrEngine.recognizeText(cropBitmap)
                        val elapsed = System.currentTimeMillis() - start
                        Pair(recResult, elapsed)
                    } finally {
                        cropBitmap.recycle()
                    }

                    val rec = result.first
                    val elapsed = result.second
                    val conf = (rec.confidence * 100).roundToInt() / 100f

                    resultText = buildString {
                        append("识别区域: [$left, $top, $right, $bottom] (${width}×${height})\n\n")
                        append("纯推理耗时: ${elapsed} ms (不含模型与图片加载)\n")
                        append("识别文本: \"${rec.text.ifEmpty { "(未识别到文字)" }}\"\n")
                        append("置信度: $conf")
                    }
                }
            } catch (e: Exception) {
                resultText = "识别失败: ${e.message ?: "未知错误"}"
            } finally {
                isRunning = false
            }
        }
    }

    Box(modifier = modifier.fillMaxSize()) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(start = 24.dp, end = 24.dp, bottom = 72.dp),
            verticalArrangement = Arrangement.Top,
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // Title displayed centered near the top
            Text(
                text = "建筑识别插件",
                style = MaterialTheme.typography.displaySmall,
                modifier = Modifier.padding(top = 80.dp)
            )
            Text(
                text = "安装本插件后，只需启动一次即可，无需其他操作",
                style = MaterialTheme.typography.bodyLarge,
                textAlign = TextAlign.Center,
                modifier = Modifier.padding(top = 32.dp)
            )

            Spacer(modifier = Modifier.height(48.dp))

            // OCR Test Button
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.Center
            ) {
                Button(
                    onClick = { runOcrTest() },
                    enabled = !isRunning
                ) {
                    Text(text = "OCR 模拟测试")
                }
            }

            if (isRunning) {
                Spacer(modifier = Modifier.height(20.dp))
                CircularProgressIndicator(
                    modifier = Modifier.height(28.dp)
                )
            }

            if (resultText.isNotEmpty()) {
                Spacer(modifier = Modifier.height(24.dp))
                Text(
                    text = resultText,
                    style = MaterialTheme.typography.bodyMedium,
                    textAlign = TextAlign.Start,
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(horizontal = 8.dp)
                )
            }
        }

        Text(
            text = "本插件仅用于游戏内文字与建筑的识别辅助，100%基于图色画面识别，全部源代码已在 Gitee 开源并采用 AGPL-3.0 协议发布，可配合任意进程使用。本插件不截取设备画面，不读取，不修改内存，不侵入，不控制任何程序，不干扰，不修改设备/软件运行环境。严禁将本插件用于任何违反法律法规的用途。若违反，使用者需自行承担所有责任，与原作者无关。",
            fontSize = 10.sp,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            textAlign = TextAlign.Center,
            modifier = Modifier
                .fillMaxWidth()
                .align(Alignment.BottomCenter)
                .padding(horizontal = 24.dp, vertical = 16.dp)
        )
    }
}
