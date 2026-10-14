package com.building.plugin.service

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.os.Build
import android.os.IBinder
import android.util.Log
import com.building.plugin.detector.BuildingDetector
import com.building.plugin.server.DetectorHttpServer

/**
 * Foreground service that hosts the detector HTTP server on port 13462.
 * Start via ADB:
 *   adb shell am start-foreground-service -n com.building.plugin/.service.DetectorService
 */
class DetectorService : Service() {

    companion object {
        private const val TAG = "DetectorService"
        private const val PORT = 13462
        private const val CHANNEL_ID = "detector_service_channel"
        private const val NOTIFICATION_ID = 1
    }

    private var httpServer: DetectorHttpServer? = null

    override fun onCreate() {
        super.onCreate()
        Log.i(TAG, "Service created")
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        Log.i(TAG, "Service starting on port $PORT")

        createNotificationChannel()
        startForeground(NOTIFICATION_ID, buildNotification())

        BuildingDetector.initialize(this)

        if (httpServer == null) {
            httpServer = DetectorHttpServer(PORT)
            httpServer!!.start()
            Log.i(TAG, "HTTP server started on port $PORT")
        }

        return START_STICKY
    }

    override fun onDestroy() {
        Log.i(TAG, "Service destroying")
        httpServer?.stop()
        httpServer = null
        BuildingDetector.clearWeights()
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            val channel = NotificationChannel(
                CHANNEL_ID,
                "识别辅助工具",
                NotificationManager.IMPORTANCE_LOW
            ).apply {
                description = "识别辅助工具"
            }
            val manager = getSystemService(NotificationManager::class.java)
            manager.createNotificationChannel(channel)
        }
    }

    private fun buildNotification(): Notification {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            Notification.Builder(this, CHANNEL_ID)
                .setContentTitle("识别辅助工具")
                .setSmallIcon(android.R.drawable.ic_menu_manage)
                .build()
        } else {
            @Suppress("DEPRECATION")
            Notification.Builder(this)
                .setContentTitle("识别辅助工具")
                .setSmallIcon(android.R.drawable.ic_menu_manage)
                .build()
        }
    }
}
