# -*- coding: utf-8 -*-
import io, csv, time, random, numpy as np, os
from collections import deque
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ===== Sensores (plyer) y FFT (scipy) =====
try:
    from plyer import magnetometer, accelerometer, barometer
    SENSORS_OK = True
    print("✅ plyer importado.")
except ImportError:
    SENSORS_OK = False
    print("⚠️ plyer no disponible. Se usará modo SIMULADO.")

try:
    from scipy import signal
    SCIPY_OK = True
except ImportError:
    SCIPY_OK = False
    print("⚠️ SciPy no disponible. La PSD/FFT no se actualizará.")

# ===== Kivy =====
from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.graphics.texture import Texture
from kivy.clock import Clock
from kivy.graphics import Color, Rectangle
from PIL import Image as PILImage

# ==================================================
# WIDGET PRINCIPAL
# ==================================================
class SensorKuh(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation='vertical', **kwargs)
        print("⚙️ Monitor K’uh v3.2 – Verificación de Sensores")

        # Paneles de gráficas
        panel = BoxLayout(orientation='vertical', size_hint_y=0.70)
        self.time_series_image = Image(size_hint_y=0.5)
        self.fft_image = Image(size_hint_y=0.5)
        panel.add_widget(self.time_series_image)
        panel.add_widget(self.fft_image)
        self.add_widget(panel)

        # IATT
        self.iatt_label = Label(text="IATT: ESPERANDO...", font_size=22, size_hint_y=0.10, color=[0,0,0,1])
        self.add_widget(self.iatt_label)

        # Estado backend + lecturas crudas
        self.status_label = Label(text="BACKEND: DESCONOCIDO | B=?, Ay=?, P=?", 
                                  font_size=16, size_hint_y=0.05)
        self.add_widget(self.status_label)

        # Botones
        botones = BoxLayout(size_hint_y=0.15, spacing=10, padding=10)
        self.btn_start = Button(text="▶️ Iniciar", font_size=22, on_press=self.iniciar)
        self.btn_stop  = Button(text="⏹️ Detener", font_size=22, on_press=self.detener)
        self.btn_save  = Button(text="💾 Guardar CSV", font_size=22, on_press=self.guardar_csv)
        for b in (self.btn_start, self.btn_stop, self.btn_save):
            botones.add_widget(b)
        self.add_widget(botones)

        # Buffers y control
        self.tiempo, self.mag, self.acc, self.pres = [], [], [], []
        self.t0 = time.time()
        self.evento = None
        self.dt = 0.25  # 4 Hz
        self.mag_history = deque(maxlen=240*10)
        self.temp_history = deque(maxlen=240*10)
        self.b_history    = deque(maxlen=240*10)

        # Detección de backend
        self.usando_reales = False
        self._real_hits = 0           # lecturas reales consecutivas
        self._need_hits = 5           # umbral para declarar "REAL"
        self.last_known_mag = 50.0    # fallback

    # -------------- Lecturas (REAL + Fallback) --------------
    def leer_magnetometro(self):
        if SENSORS_OK:
            try:
                val = magnetometer.magnetic_field
                if val is not None and len(val) == 3:
                    B_total = float(np.sqrt(val[0]**2 + val[1]**2 + val[2]**2))
                    self.last_known_mag = B_total
                    self._mark_hit(True)
                    return B_total
            except Exception as e:
                # print("Mag error:", e)
                pass
        self._mark_hit(False)
        # SIMULADO: oscilación lenta + ruido pequeño sobre el último valor
        return self.last_known_mag + 0.5*np.sin(2*np.pi*0.08*time.time()) + random.uniform(-0.5, 0.5)

    def leer_acelerometro(self):
        if SENSORS_OK:
            try:
                val = accelerometer.acceleration
                if val is not None and len(val) >= 2 and val[1] is not None:
                    return float(val[1])
            except: 
                pass
        return 9.8 + 0.3*np.cos(time.time()*1.1) + random.uniform(-0.1, 0.1)

    def leer_barometro(self):
        if SENSORS_OK:
            try:
                val = barometer.pressure
                if val is not None:
                    return float(val)
            except:
                pass
        return 1013 + 0.5*np.sin(2*np.pi*0.05*time.time()) + random.uniform(-0.2, 0.2)

    def _mark_hit(self, real_ok: bool):
        # Cuenta impactos reales consecutivos para declarar "REAL"
        if real_ok:
            self._real_hits += 1
            if self._real_hits >= self._need_hits:
                self.usando_reales = True
        else:
            # si falla, resetea el contador; si hay varias fallas vuelve a SIMULADO
            self._real_hits = 0
            # opcional: si ya estaba en REAL y falla mucho, baja a SIMULADO
            # (aquí con 8 fallas seguidas)
            if self.usando_reales:
                pass

    # -------------- Control --------------
    def iniciar(self, *args):
        if self.evento: 
            return
        if SENSORS_OK:
            try:
                magnetometer.enable()
            except: pass
            try:
                accelerometer.enable()
            except: pass
            try:
                barometer.enable()
            except: pass
            print("🔌 Intento de habilitar sensores físicos (plyer).")
        self._render_blank_image(self.time_series_image)
        self._render_blank_image(self.fft_image)
        self.evento = Clock.schedule_interval(self.actualizar, self.dt)
        self.t0 = time.time()
        self.tiempo, self.mag, self.acc, self.pres = [], [], [], []
        self.mag_history.clear(); self.temp_history.clear(); self.b_history.clear()
        self.usando_reales = False; self._real_hits = 0
        print("📡 Medición iniciada a 4 Hz.")

    def detener(self, *args):
        if not self.evento: 
            return
        self.evento.cancel()
        self.evento = None
        if SENSORS_OK:
            try: magnetometer.disable()
            except: pass
            try: accelerometer.disable()
            except: pass
            try: barometer.disable()
            except: pass
        print("🛑 Medición detenida.")

    # -------------- IATT --------------
    def robust_zscore(self, serie):
        if len(serie) < 240*5: return 0.0
        arr = np.array(serie, dtype=float)
        m = np.median(arr)
        mad = np.median(np.abs(arr - m))
        if mad == 0: return 0.0
        return (arr[-1] - m) / (1.4826 * mad)

    def spectral_persistence_index(self, serie):
        if (not SCIPY_OK) or len(serie) < 240: return 0.5
        Fs = 1.0/self.dt
        data = np.array(serie, dtype=float)
        nperseg = min(int(Fs*10), len(data))
        f, Pxx = signal.welch(data, Fs, nperseg=nperseg, noverlap=nperseg//2)
        idx = (f >= 0.02) & (f <= 0.2)
        if not np.any(idx): return 0.5
        ruido = np.median(Pxx[f > 1.0]) if np.any(f > 1.0) else np.median(Pxx)
        if ruido == 0: return 0.5
        snr = np.max(Pxx[idx]) / ruido
        return float(np.clip((snr - 1)/5, 0, 1))

    def soliton_rate_detector(self):
        if len(self.mag_history) < 240*2: return 0.0
        arr = np.array(self.mag_history, dtype=float)
        umbral = np.median(arr) * 1.05
        tasa = np.sum(arr > umbral) / (240*1.0)
        return float(np.clip(tasa, 0, 1))

    def calcular_iatt_pro(self):
        zr  = self.robust_zscore(self.mag_history)
        spi = self.spectral_persistence_index(self.mag_history)
        sr  = self.soliton_rate_detector()
        zrN  = np.clip(zr/5, 0, 1)
        spiN = 1 - spi
        srN  = np.clip(sr*2, 0, 1)
        iatt = 0.5*zrN + 0.3*spiN + 0.2*srN

        if iatt >= 0.65 or zrN >= 0.9: color, estado = [1,0,0,1], "ROJA 🔴"
        elif iatt >= 0.40:             color, estado = [1,1,0,1], "AMARILLA 🟡"
        else:                           color, estado = [0,1,0,1], "VERDE 🟢"

        self.iatt_label.text = f"IATT: {iatt:.2f} - {estado}"
        self.iatt_label.color = [0,0,0,1]
        self.iatt_label.canvas.before.clear()
        with self.iatt_label.canvas.before:
            Color(*color)
            Rectangle(pos=self.iatt_label.pos, size=self.iatt_label.size)

    # -------------- Render helpers --------------
    def _render_to_texture(self, fig, image_widget):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
        buf.seek(0)
        im = PILImage.open(buf)
        tex = Texture.create(size=im.size, colorfmt='rgba')
        tex.blit_buffer(im.convert('RGBA').tobytes(), colorfmt='rgba', bufferfmt='ubyte')
        tex.flip_vertical()
        image_widget.texture = tex

    def _render_blank_image(self, image_widget):
        im = PILImage.new('RGB', (100, 100), color='black')
        buf = io.BytesIO(); im.save(buf, format='png'); buf.seek(0)
        im = PILImage.open(buf)
        tex = Texture.create(size=im.size, colorfmt='rgb')
        tex.blit_buffer(im.tobytes(), colorfmt='rgb', bufferfmt='ubyte')
        tex.flip_vertical()
        image_widget.texture = tex

    # -------------- FFT & Loop --------------
    def render_fft(self):
        if len(self.mag) < 100 or not SCIPY_OK: 
            return
        Fs = 1.0/self.dt
        f, Pxx = signal.welch(np.array(self.mag, dtype=float), Fs, nperseg=64, noverlap=32)
        fig, ax = plt.subplots(figsize=(5.8, 2.0), dpi=120)
        m = f <= 1.0
        ax.plot(f[m], Pxx[m], 'r-', lw=1.3)
        ax.axvspan(0.02, 0.2, color='yellow', alpha=0.3, label='Banda ωR–M')
        ax.set_title("⚡ PSD: Búsqueda de ωR–M", fontsize=11)
        ax.set_xlabel("Frecuencia (Hz)", fontsize=9)
        ax.set_yscale('log')
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(fontsize=8, loc='upper right')
        self._render_to_texture(fig, self.fft_image)

    def actualizar(self, dt):
        t = time.time() - self.t0
        B  = self.leer_magnetometro()
        Ay = self.leer_acelerometro()
        P  = self.leer_barometro()

        self.tiempo.append(t); self.mag.append(B); self.acc.append(Ay); self.pres.append(P)
        self.mag_history.append(B); self.temp_history.append(0); self.b_history.append(0)

        if len(self.tiempo) > 240:
            self.tiempo, self.mag, self.acc, self.pres = (
                self.tiempo[-240:], self.mag[-240:], self.acc[-240:], self.pres[-240:]
            )

        # Serie de tiempo
        fig, ax = plt.subplots(figsize=(5.8, 2.0), dpi=120)
        def norm(d):
            arr = np.array(d, dtype=float)
            rng = np.max(arr) - np.min(arr)
            return (arr - np.min(arr)) / (rng + 1e-6) * 10.0
        if self.tiempo:
            ax.plot(self.tiempo, norm(self.mag), 'r-', lw=1.3, label='Magnético (L)')
            ax.plot(self.tiempo, norm(self.acc), 'g-', lw=1.3, label='Acelerómetro Y')
            ax.plot(self.tiempo, norm(self.pres), 'b-', lw=1.3, label='Presión')
            ax.set_xlim(max(0, self.tiempo[-1]-60), self.tiempo[-1])
            ax.set_ylim(-1, 11)
        ax.set_title("📈 Serie de Tiempo (60s)", fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right'); ax.grid(True, linestyle='--', alpha=0.5)
        self._render_to_texture(fig, self.time_series_image)

        # FFT + IATT
        self.render_fft()
        self.calcular_iatt_pro()

        # Actualiza banner de backend + lecturas crudas
        backend = "REAL" if self.usando_reales else "SIMULADO"
        self.status_label.text = f"BACKEND: {backend} | B={B:.2f}  Ay={Ay:.2f}  P={P:.2f}"
        # Log útil:
        # print(self.status_label.text)

    # -------------- Guardado --------------
    def guardar_csv(self, *args):
        backend = "REAL" if self.usando_reales else "SIMULADO"
        nombre = f"/storage/emulated/0/Download/datos_kuh_v32_{backend.lower()}_{int(time.time())}.csv"
        with open(nombre, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(["Tiempo","Magnetico_B","Acelerometro_Y","Presion_hPa","backend"])
            for t, m, a, p in zip(self.tiempo, self.mag, self.acc, self.pres):
                w.writerow([t, m, a, p, backend])
        print(f"✅ Datos guardados en {nombre}")

# ==================================================
class SensorKuhApp(App):
    def build(self):
        return SensorKuh()

if __name__ == '__main__':
    from kivy.core.window import Window
    Window.size = (1080, 2250)
    SensorKuhApp().run()