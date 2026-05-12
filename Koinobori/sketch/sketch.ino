#include <Arduino.h>

// ==========================================
// 1. 風向センサー管理クラス (Wokwi用モック: ポテンショメータ)
// ※実機が届いたら、中身をAS5600(I2C)用に書き換えます
// ==========================================
class WindDirectionSensor {
private:
    uint8_t pin;
    float currentAngle;

public:
    WindDirectionSensor(uint8_t analog_pin) : pin(analog_pin), currentAngle(0.0) {}

    void begin() {
        analogReadResolution(12); // ESP32のADC解像度を12bit(0-4095)に設定
        pinMode(pin, INPUT);
    }

    // 裏で動くタスクから定期的に呼ばれる更新処理
    void update() {
        int rawValue = analogRead(pin);
        currentAngle = (rawValue / 4095.0) * 360.0; // 0-4095を0-360度に変換
        if (currentAngle >= 360.0) currentAngle = 0.0;
    }

    float getAngle() { 
        return currentAngle; 
    }

    // 角度を方位文字列(N, NE, E...)に変換
    String getDirectionString() {
        const char* directions[] = {"N", "NE", "E", "SE", "S", "SW", "W", "NW", "N"};
        int index = round(currentAngle / 45.0);
        return String(directions[index % 8]);
    }
};

// ==========================================
// 2. 風速センサー管理クラス (ハードウェア割り込み)
// ※実機が届いたら、このままA3144(ホールセンサ)に繋げば動きます
// ==========================================
class WindSpeedSensor {
private:
    uint8_t pin;
    volatile uint32_t pulseCount;
    volatile uint32_t lastInterruptTime;
    const uint32_t DEBOUNCE_DELAY_MS = 10; // 10ms以内の連続反応(チャタリング)は無視

    // 割り込みサービスルーチン (メモリの高速領域に配置)
    static void IRAM_ATTR isrHandler(void* arg) {
        WindSpeedSensor* sensor = static_cast<WindSpeedSensor*>(arg);
        uint32_t currentTime = millis();
        
        // ソフトウェア・デバウンス
        if (currentTime - sensor->lastInterruptTime > sensor->DEBOUNCE_DELAY_MS) {
            sensor->pulseCount++;
            sensor->lastInterruptTime = currentTime;
        }
    }

public:
    WindSpeedSensor(uint8_t gpio_pin) : pin(gpio_pin), pulseCount(0), lastInterruptTime(0) {}

    void begin() {
        pinMode(pin, INPUT_PULLUP); // 内部プルアップ有効
        // ピンの電圧がHIGHからLOWに落ちた(FALLING)瞬間に割り込みを発生させる
        attachInterruptArg(digitalPinToInterrupt(pin), isrHandler, this, FALLING); 
    }

    // 蓄積されたパルス数を取得し、ゼロにリセットする
    uint32_t getAndResetPulseCount() {
        noInterrupts(); // データ取得中の割り込みを一時停止（データ破損防止）
        uint32_t currentCount = pulseCount;
        pulseCount = 0;
        interrupts();
        return currentCount;
    }
};

// ==========================================
// グローバル変数・インスタンス
// ==========================================
WindDirectionSensor windDirSensor(34); // 風向ツマミは GPIO 34
WindSpeedSensor windSpeedSensor(32);   // 風速ボタンは GPIO 32

uint32_t lastOutputTime = 0;
const uint32_t OUTPUT_INTERVAL_MS = 2000; // 2秒ごとに画面に出力

// ==========================================
// FreeRTOS タスク: 風向の常時監視（裏で動き続けるスレッド）
// ==========================================
void windDirectionTask(void *pvParameters) {
    while (true) {
        windDirSensor.update();
        // 100msごとに実行し、その間は他の処理(Wi-Fi等)にCPUを譲る
        vTaskDelay(pdMS_TO_TICKS(100)); 
    }
}

// ==========================================
// メイン処理 (setup & loop)
// ==========================================
void setup() {
    Serial.begin(115200);
    
    windDirSensor.begin();
    windSpeedSensor.begin();

    // デュアルコアの強み: Core 1 に風向監視専用のタスクを立ち上げる
    xTaskCreatePinnedToCore(
        windDirectionTask, "WindDirTask", 2048, NULL, 1, NULL, 1
    );

    Serial.println("===============================");
    Serial.println("Kazamidori Anemometer OS Booted!");
    Serial.println("===============================");
}

void loop() {
    uint32_t currentTime = millis();

    // 2秒ごとに風速と風向を集計してシリアルモニタに出力
    if (currentTime - lastOutputTime >= OUTPUT_INTERVAL_MS) {
        lastOutputTime = currentTime;

        // 風速（パルス数）を取得してリセット
        uint32_t pulses = windSpeedSensor.getAndResetPulseCount();
        
        // 風向を取得
        float angle = windDirSensor.getAngle();
        String dirStr = windDirSensor.getDirectionString();

        // 画面に出力
        Serial.printf("Wind Pulses (last 2s): %d | Direction: %.1f deg [%s]\n", pulses, angle, dirStr.c_str());
    }
}