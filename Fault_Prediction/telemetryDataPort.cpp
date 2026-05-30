void setup() {
  Serial.begin(115200); // High-speed data stream
}

void loop() {
    int rawValue = analogRead(A0); // Read the current tap voltage
    float voltage = rawValue * (5.0 / 1023.0); // Convert to actual volts (assuming 5V logic)

  // Because R = 1 Ohm, Voltage Drop = Current Draw (Amps)
    float currentAmps = voltage; 

  // Format as a clear timestamped or sequential stream
    Serial.print(millis());
    Serial.print(",");
    Serial.println(currentAmps);

  delay(10); // Log data points every 10 milliseconds
}
