package com.building.plugin

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Color
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
import androidx.compose.foundation.layout.width
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
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.building.plugin.detector.BuildingDetector
import com.building.plugin.service.DetectorService
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlin.random.Random


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
    // State for the simulated detection run
    var resultText by remember { mutableStateOf("") }
    var isRunning by remember { mutableStateOf(false) }
    val scope = rememberCoroutineScope()

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

            // Simulated detection button
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.Center
            ) {
                Button(
                    onClick = {
                        if (isRunning) return@Button
                        scope.launch {
                            isRunning = true
                            resultText = "正在加载模型并执行检测…"
                            try {
                                withContext(Dispatchers.Default) {
                                    // Load the main-base-battle model
                                    BuildingDetector.loadWeights("main-base-battle")

                                    // Generate 3 random 640x640 bitmaps
                                    val bitmaps = (1..3).map {
                                        Bitmap.createBitmap(
                                            640, 640, Bitmap.Config.ARGB_8888
                                        ).apply {
                                            eraseColor(
                                                Color.rgb(
                                                    Random.nextInt(256),
                                                    Random.nextInt(256),
                                                    Random.nextInt(256)
                                                )
                                            )
                                        }
                                    }

                                    // Run detection sequentially and measure total time
                                    val start = System.currentTimeMillis()
                                    for (bmp in bitmaps) {
                                        BuildingDetector.detect(
                                            bitmap = bmp,
                                            clearWeightsAfter = false
                                        )
                                    }
                                    val elapsed = System.currentTimeMillis() - start

                                    // Free the generated bitmaps
                                    bitmaps.forEach { it.recycle() }

                                    resultText = "检测完成，总耗时：${elapsed} 毫秒"
                                }
                            } catch (e: Exception) {
                                resultText = "检测失败：${e.message ?: "未知错误"}"
                            } finally {
                                isRunning = false
                            }
                        }
                    },
                    enabled = !isRunning
                ) {
                    Text(text = "模拟检测")
                }

                if (isRunning) {
                    Spacer(modifier = Modifier.width(16.dp))
                    CircularProgressIndicator(
                        modifier = Modifier.height(24.dp)
                    )
                }
            }

            if (resultText.isNotEmpty()) {
                Spacer(modifier = Modifier.height(24.dp))
                Text(
                    text = resultText,
                    style = MaterialTheme.typography.bodyLarge,
                    textAlign = TextAlign.Center
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
