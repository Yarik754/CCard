import sys
import time
import threading
import psutil
import json
import re
import unicodedata
import numpy as np
import cv2
import sounddevice as sd
from pygrabber.dshow_graph import FilterGraph

from PyQt6 import QtCore, QtGui, QtWidgets

try:
    import win32con
    import win32gui
    WIN32_AVAILABLE = True
except Exception:
    WIN32_AVAILABLE = False


def load_settings(filename="settings.json"):
    """Download settings JSON"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading settings from {filename}: {e}")
        return {
            "resolution": [1280, 720],
            "fps": 120
        }


def save_settings(settings, filename="settings.json"):
    """Save settings JSON"""
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(settings, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"Error saving settings to {filename}: {e}")


# Download settings
SETTINGS = load_settings()
WIDTH, HEIGHT = 1280, 720
CONTROL_PANEL_HEIGHT = 80
DEVICE_POLL_INTERVAL = 2.0

# Translateions
TRANSLATIONS = {
    "uk": {
        "camera": "Камера",
        "resolution": "Розділення",
        "fps": "FPS",
        "audio_in": "Аудіо вхід",
        "toggle_debug": "Дебаг режим",
        "fullscreen": "На весь екран",
        "settings": "Налаштування",
        "settings_title": "Налаштування",
        "theme": "Тема",
        "language": "Мова",
        "dark": "Темна",
        "light": "Світла",
        "ok": "ОК",
        "cancel": "Скасувати",
    },
    "en": {
        "camera": "Camera",
        "resolution": "Resolution",
        "fps": "FPS",
        "audio_in": "Audio In",
        "toggle_debug": "Toggle Debug",
        "fullscreen": "Fullscreen",
        "settings": "Settings",
        "settings_title": "Settings",
        "theme": "Theme",
        "language": "Language",
        "dark": "Dark",
        "light": "Light",
        "ok": "OK",
        "cancel": "Cancel",
    }
}

# Get text for current language

def get_text(key):
    lang = SETTINGS.get("language", "uk")
    return TRANSLATIONS.get(lang, TRANSLATIONS["uk"]).get(key, key)

# Apply theme

def apply_theme():
    theme = SETTINGS.get("theme", "dark")
    if theme == "dark":
        style = """
        QWidget { background-color: #1e1e1e; color: white; }
        QComboBox { background-color: #2e2e2e; color: white; border: 1px solid #555; padding: 3px; }
        QComboBox::drop-down { border: none; }
        QPushButton { background-color: #3e3e3e; color: white; border: 1px solid #555; padding: 5px; }
        QPushButton:hover { background-color: #4e4e4e; border: 1px solid #777; }
        QLabel { color: white; background-color: transparent; }
        QDialog { background-color: #1e1e1e; }
        """
    else:
        style = """
        QWidget { background-color: #ffffff; color: black; }
        QComboBox { background-color: #f5f5f5; color: black; border: 1px solid #ccc; padding: 3px; }
        QComboBox::drop-down { border: none; }
        QPushButton { background-color: #e0e0e0; color: black; border: 1px solid #999; padding: 5px; }
        QPushButton:hover { background-color: #d0d0d0; border: 1px solid #666; }
        QLabel { color: black; background-color: transparent; }
        QDialog { background-color: #ffffff; }
        """
    return style

# Get video devices from pygrabber

def list_video_devices():
    try:
        g = FilterGraph()
        devices = g.get_input_devices()
        if not devices:
            return []
        return devices
    except Exception as e:
        print("Error listing video devices:", e)
        return []

# Get audio devices from sounddevice

def list_audio_input_devices():
    try:
        devices = sd.query_devices()
        inputs = []
        seen_norm = set()
        for d in devices:
            try:
                if d.get('max_input_channels', 0) > 0:
                    name = d.get('name')
                    if not name:
                        continue
                    # Unicode normalize, remove punctuation, collapse spaces, lowercase
                    norm = unicodedata.normalize('NFKC', name)
                    norm = norm.strip().lower()
                    # remove punctuation (keep word chars and spaces)
                    norm = re.sub(r"[^\w\s]", "", norm)
                    # collapse multiple spaces
                    norm = re.sub(r"\s+", " ", norm).strip()
                    if norm and norm not in seen_norm:
                        inputs.append(name)
                        seen_norm.add(norm)
            except Exception:
                continue
        return inputs
    except Exception as e:
        print("Error listing audio devices:", e)
        return []


class CameraThread(QtCore.QThread):
    frame_signal = QtCore.pyqtSignal(np.ndarray)
    fps_signal = QtCore.pyqtSignal(float)

    def __init__(self, cam_index=0, resolution=(1280,720), fps=120, parent=None):
        super().__init__(parent)
        self.cam_index = cam_index
        self.resolution = resolution
        self.fps = fps
        self._running = False
        self.cap = None

    def run(self):
        self._running = True
        try:
            self.cap = cv2.VideoCapture(self.cam_index, cv2.CAP_DSHOW)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.resolution[0]))
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.resolution[1]))
            self.cap.set(cv2.CAP_PROP_FPS, int(self.fps))
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception as e:
            print("Camera open exception:", e)
            self._running = False
            return

        prev = time.time()
        while self._running:
            ret, img = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            now = time.time()
            fps_val = 1.0 / (now - prev) if (now - prev) > 1e-6 else 0.0
            prev = now
            self.fps_signal.emit(fps_val)
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.frame_signal.emit(rgb)

            time.sleep(0.001)

        try:
            if self.cap:
                self.cap.release()
        except Exception:
            pass

    def stop(self):
        self._running = False
        self.wait(1000)


class AudioWorker(threading.Thread):
    def __init__(self, input_device_index=None):
        super().__init__(daemon=True)
        self.input_device_index = input_device_index
        self._running = threading.Event()
        self._running.clear()

    def run(self):
        def callback(indata, outdata, frames, time_info, status):
            if status:
                print("Audio status:", status)
            try:
                outdata[:] = indata
            except Exception:
                outdata.fill(0)

        try:
            with sd.Stream(device=(self.input_device_index, None),
                           channels=1, samplerate=44100, callback=callback):
                self._running.set()
                while self._running.is_set():
                    sd.sleep(100)
        except Exception as e:
            print("Audio worker error:", e)

    def stop(self):
        self._running.clear()

class SettingsDialog(QtWidgets.QDialog):
    settings_changed = QtCore.pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle(get_text("settings_title"))
        self.setModal(True)
        self.setGeometry(100, 100, 450, 320)

        layout = QtWidgets.QVBoxLayout(self)

        # Theme
        theme_layout = QtWidgets.QHBoxLayout()
        theme_layout.addWidget(QtWidgets.QLabel(get_text("theme")))
        self.theme_combo = QtWidgets.QComboBox()
        self.theme_combo.addItems([get_text("dark"), get_text("light")])
        current_theme = SETTINGS.get("theme", "dark")
        self.theme_combo.setCurrentText(get_text(current_theme))
        theme_layout.addWidget(self.theme_combo)
        layout.addLayout(theme_layout)

        # Language
        lang_layout = QtWidgets.QHBoxLayout()
        lang_layout.addWidget(QtWidgets.QLabel(get_text("language")))
        self.language_combo = QtWidgets.QComboBox()
        self.language_combo.addItems(["Українська", "English"])
        current_lang = SETTINGS.get("language", "uk")
        self.language_combo.setCurrentIndex(0 if current_lang == "uk" else 1)
        lang_layout.addWidget(self.language_combo)
        layout.addLayout(lang_layout)

        # Resolution
        res_layout = QtWidgets.QHBoxLayout()
        res_layout.addWidget(QtWidgets.QLabel(get_text("resolution")))
        self.res_combo = QtWidgets.QComboBox()
        self.res_combo.addItems(["640x480", "1280x720", "1920x1080"])
        current_res = SETTINGS.get("resolution", [1280, 720])
        self.res_combo.setCurrentText(f"{current_res[0]}x{current_res[1]}")
        res_layout.addWidget(self.res_combo)
        layout.addLayout(res_layout)

        # FPS
        fps_layout = QtWidgets.QHBoxLayout()
        fps_layout.addWidget(QtWidgets.QLabel(get_text("fps")))
        self.fps_combo = QtWidgets.QComboBox()
        self.fps_combo.addItems(["10", "30", "60", "120"])
        current_fps = str(SETTINGS.get("fps", 120))
        self.fps_combo.setCurrentText(current_fps)
        fps_layout.addWidget(self.fps_combo)
        layout.addLayout(fps_layout)

        layout.addStretch()

        # Buttons
        button_layout = QtWidgets.QHBoxLayout()
        ok_btn = QtWidgets.QPushButton(get_text("ok"))
        cancel_btn = QtWidgets.QPushButton(get_text("cancel"))
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        button_layout.addWidget(ok_btn)
        button_layout.addWidget(cancel_btn)
        layout.addLayout(button_layout)

# Get settings

    def get_settings(self):
        theme_text = self.theme_combo.currentText()
        theme = "dark" if "dark" in theme_text.lower() or "темна" in theme_text.lower() else "light"
        
        lang_idx = self.language_combo.currentIndex()
        language = "uk" if lang_idx == 0 else "en"
        
        res_text = self.res_combo.currentText()
        w, h = map(int, res_text.split("x"))
        resolution = [w, h]
        
        fps = int(self.fps_combo.currentText())
        
        return {
            "theme": theme,
            "language": language,
            "resolution": resolution,
            "fps": fps
        }

class FullscreenVideoWindow(QtWidgets.QWidget):
    exit_fullscreen = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowFlags(QtCore.Qt.WindowType.FramelessWindowHint)
        self.setAttribute(QtCore.Qt.WidgetAttribute.WA_ShowWithoutActivating, False)
        self.video_label = QtWidgets.QLabel(self)
        self.video_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.debug_label = QtWidgets.QLabel(self)
        self.debug_label.setStyleSheet("color: white; background: rgba(0,0,0,0);")
        self.debug_label.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.addWidget(self.video_label)
        self._current_pixmap = None

    def set_frame(self, frame: np.ndarray):
        if frame is None:
            return
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(frame.data.tobytes(), w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qimg).scaled(self.width(), self.height(), QtCore.Qt.AspectRatioMode.IgnoreAspectRatio)
        self.video_label.setPixmap(pix)
        self._current_pixmap = pix

    def set_debug_text(self, text: str):
        self.debug_label.setText(text)
        self.debug_label.adjustSize()
        self.debug_label.move(10, 10)
        if not self.debug_label.isVisible():
            self.debug_label.show()

    def showFullScreen(self):
        super().showFullScreen()
        self.video_label.setFixedSize(self.size())
        self.debug_label.setParent(self)
        self.debug_label.raise_()

    def keyPressEvent(self, event):
        self.exit_fullscreen.emit()

    def mousePressEvent(self, event):
        self.exit_fullscreen.emit()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if self._current_pixmap:
            self.video_label.setFixedSize(self.size())
            self.set_frame(np.array(self._current_pixmap.toImage().bits()).reshape((self._current_pixmap.height(), self._current_pixmap.width(), 4)) if False else None)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("CCard ver 1.2.1")
        self.resize(WIDTH, HEIGHT)
        self.setMinimumSize(1024, 768)

        self.debug_mode = True
        self.current_frame = None
        self.video_fps = 0.0
        self.cpu = 0.0
        self.ram_mb = 0.0

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QVBoxLayout(central)
        main_layout.setContentsMargins(0,0,0,0)
        main_layout.setSpacing(0)

        self.video_area = QtWidgets.QWidget()
        self.video_area_layout = QtWidgets.QVBoxLayout(self.video_area)
        self.video_area_layout.setContentsMargins(0,0,0,0)
        self.video_area_layout.setSpacing(0)
        main_layout.addWidget(self.video_area, 1)

        self.video_label = QtWidgets.QLabel()
        self.video_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_area_layout.addWidget(self.video_label)

        self.debug_label = QtWidgets.QLabel(self.video_area)
        self.debug_label.setStyleSheet("color: white; background: rgba(0,0,0,0);")
        self.debug_label.setAttribute(QtCore.Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.debug_label.move(10, 10)
        self.debug_label.raise_()

        self.controls = QtWidgets.QWidget()
        self.controls.setFixedHeight(CONTROL_PANEL_HEIGHT)
        controls_layout = QtWidgets.QHBoxLayout(self.controls)
        controls_layout.setContentsMargins(6,6,6,6)
        controls_layout.setSpacing(8)
        main_layout.addWidget(self.controls)

        self.video_devices = list_video_devices()
        self.audio_devices = list_audio_input_devices()
        # Extra safety: ensure unique audio device names (normalize)
        try:
            unique_audio = []
            seen_norm_local = set()
            for name in self.audio_devices:
                if not name:
                    continue
                norm = unicodedata.normalize('NFKC', name).strip().lower()
                norm = re.sub(r"[^\w\s]", "", norm)
                norm = re.sub(r"\s+", " ", norm).strip()
                if norm and norm not in seen_norm_local:
                    unique_audio.append(name)
                    seen_norm_local.add(norm)
            self.audio_devices = unique_audio
        except Exception:
            pass

        self.camera_label = QtWidgets.QLabel(get_text("camera"))
        controls_layout.addWidget(self.camera_label)
        self.camera_combo = QtWidgets.QComboBox()
        self.camera_combo.addItems(self.video_devices if self.video_devices else ["No Devices"])
        controls_layout.addWidget(self.camera_combo)

        controls_layout.addStretch()
        self.audio_label = QtWidgets.QLabel(get_text("audio_in"))
        controls_layout.addWidget(self.audio_label)
        self.audio_combo = QtWidgets.QComboBox()
        self.audio_combo.addItems(self.audio_devices if self.audio_devices else ["No Audio Input Devices"])
        controls_layout.addWidget(self.audio_combo)

        self.debug_btn = QtWidgets.QPushButton(get_text("toggle_debug"))
        self.fullscreen_btn = QtWidgets.QPushButton(get_text("fullscreen"))
        self.settings_btn = QtWidgets.QPushButton(get_text("settings"))
        controls_layout.addWidget(self.debug_btn)
        controls_layout.addWidget(self.fullscreen_btn)
        controls_layout.addWidget(self.settings_btn)

        self.camera_combo.currentIndexChanged.connect(self.on_camera_changed)
        self.audio_combo.currentIndexChanged.connect(self.on_audio_changed)
        self.debug_btn.clicked.connect(self.toggle_debug)
        self.fullscreen_btn.clicked.connect(self.open_fullscreen)
        self.settings_btn.clicked.connect(self.open_settings)

        self.camera_thread = None
        self.audio_worker = None

        self.fullscreen_window = FullscreenVideoWindow()
        self.fullscreen_window.exit_fullscreen.connect(self.close_fullscreen)

        self.ui_timer = QtCore.QTimer(self)
        self.ui_timer.setInterval(500)  # ms
        self.ui_timer.timeout.connect(self._update_stats_and_overlay)
        self.ui_timer.start()



        res = SETTINGS.get("resolution", [1280, 720])
        fps = SETTINGS.get("fps", 120)
        self._start_camera(index=0, resolution=tuple(res), fps=fps)
        self._start_default_audio()



        self.setStyleSheet(apply_theme())
        self.fullscreen_window.setStyleSheet(apply_theme())

        if WIN32_AVAILABLE:
            try:
                self._start_win32_device_watcher()
            except Exception as e:
                print("win32 watcher start failed, fallback to polling:", e)
                self._start_device_poller()
        else:
            self._start_device_poller()

    def _start_camera(self, index=0, resolution=(1280,720), fps=120):
        if self.camera_thread and self.camera_thread.isRunning():
            try:
                self.camera_thread.stop()
            except Exception:
                pass
        self.camera_thread = CameraThread(cam_index=index, resolution=resolution, fps=fps)
        self.camera_thread.frame_signal.connect(self._on_frame)
        self.camera_thread.fps_signal.connect(self._on_cam_fps)
        self.camera_thread.start()

    def _on_frame(self, frame: np.ndarray):
        self.current_frame = frame
        if frame is None:
            return
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(frame.data.tobytes(), w, h, bytes_per_line, QtGui.QImage.Format.Format_RGB888)
        target_w = self.video_label.width()
        target_h = self.video_label.height()
        pix = QtGui.QPixmap.fromImage(qimg).scaled(target_w, target_h, QtCore.Qt.AspectRatioMode.IgnoreAspectRatio)
        self.video_label.setPixmap(pix)

        if self.fullscreen_window.isVisible():
            self.fullscreen_window.set_frame(frame)

    def _on_cam_fps(self, fps):
        self.video_fps = fps

    def _start_default_audio(self):
        audio_list = list_audio_input_devices()
        if audio_list:
            try:
                devices = sd.query_devices()
                for i, d in enumerate(devices):
                    if d['max_input_channels'] > 0:
                        sd.default.device = (i, None)
                        self._start_audio(i)
                        break
            except Exception as e:
                print("start default audio error:", e)

    def _start_audio(self, device_index):
        if self.audio_worker:
            try:
                self.audio_worker.stop()
            except Exception:
                pass
        self.audio_worker = AudioWorker(device_index)
        self.audio_worker.start()

    def on_camera_changed(self, idx):

        print("Camera changed ->", idx)
        try:
            res = SETTINGS.get("resolution", [1280, 720])
            fps = SETTINGS.get("fps", 120)
            self._start_camera(index=idx, resolution=tuple(res), fps=fps)
        except Exception as e:
            print("on_camera_changed error:", e)

    def on_audio_changed(self, idx):
        name = self.audio_combo.currentText()
        print("Audio changed ->", name)
        try:
            devices = sd.query_devices()
            found = None
            for i, d in enumerate(devices):
                if d['name'] == name and d['max_input_channels'] > 0:
                    found = i
                    break
            if found is not None:
                sd.default.device = (found, None)
                self._start_audio(found)
        except Exception as e:
            print("on_audio_changed error:", e)

    def toggle_debug(self):
        self.debug_mode = not self.debug_mode
        if not self.debug_mode:
            self.debug_label.hide()
            self.fullscreen_window.debug_label.hide()
        else:
            self.debug_label.show()
            self.fullscreen_window.debug_label.show()

    def open_fullscreen(self):
        self.fullscreen_window.resize(QtWidgets.QApplication.primaryScreen().size())
        if self.current_frame is not None:
            self.fullscreen_window.set_frame(self.current_frame)
        if self.debug_mode:
            self.fullscreen_window.set_debug_text(self._compose_debug_text())
        self.fullscreen_window.showFullScreen()

    def close_fullscreen(self):
        self.fullscreen_window.hide()

    def _start_device_poller(self):
        self._device_poller_stop = threading.Event()
        self._device_poller_stop.clear()
        def poller_loop():
            prev_v = None
            prev_a = None
            while not self._device_poller_stop.is_set():
                try:
                    v = list_video_devices()
                    a = list_audio_input_devices()
                    if v != prev_v or a != prev_a:
                        prev_v = v
                        prev_a = a
                        QtCore.QMetaObject.invokeMethod(self, "_update_device_lists", QtCore.Qt.ConnectionType.QueuedConnection,
                                                        QtCore.Q_ARG(list, v), QtCore.Q_ARG(list, a))
                except Exception as e:
                    print("Device poller error:", e)
                time.sleep(DEVICE_POLL_INTERVAL)
        self._poller_thread = threading.Thread(target=poller_loop, daemon=True)
        self._poller_thread.start()

    def _start_win32_device_watcher(self):
        if not WIN32_AVAILABLE:
            raise RuntimeError("pywin32 not available")

        def wndproc(hwnd, msg, wparam, lparam):
            if msg == win32con.WM_DEVICECHANGE:
                v = list_video_devices()
                a = list_audio_input_devices()
                QtCore.QMetaObject.invokeMethod(self, "_update_device_lists", QtCore.Qt.ConnectionType.QueuedConnection,
                                                QtCore.Q_ARG(list, v), QtCore.Q_ARG(list, a))
            return True

        message_map = {win32con.WM_DEVICECHANGE: wndproc}
        wc = win32gui.WNDCLASS()
        hinst = wc.hInstance = win32gui.GetModuleHandle(None)
        wc.lpszClassName = "CCardDeviceWatcher"
        wc.lpfnWndProc = wndproc
        class_atom = win32gui.RegisterClass(wc)
        hwnd = win32gui.CreateWindowEx(0, class_atom, "CCardWatcher", 0, 0, 0, 0, 0, 0, 0, hinst, None)
        def message_loop():
            try:
                while True:
                    win32gui.PumpWaitingMessages()
                    time.sleep(0.1)
            except Exception as e:
                print("win32 message loop terminated:", e)
        self._win32_msg_thread = threading.Thread(target=message_loop, daemon=True)
        self._win32_msg_thread.start()

    @QtCore.pyqtSlot(list, list)
    def _update_device_lists(self, video_list, audio_list):
        cur_video = self.camera_combo.currentText()
        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        if video_list:
            self.camera_combo.addItems(video_list)
        else:
            self.camera_combo.addItem("No Devices")
        idx = -1
        try:
            idx = self.camera_combo.findText(cur_video)
        except Exception:
            idx = -1
        if idx != -1:
            self.camera_combo.setCurrentIndex(idx)
        else:
            self.camera_combo.setCurrentIndex(0)
        self.camera_combo.blockSignals(False)

        # Audio
        cur_audio = self.audio_combo.currentText()
        self.audio_combo.blockSignals(True)
        self.audio_combo.clear()
        if audio_list:
            # Ensure audio_list is deduplicated here too (extra safety)
            unique_audio = []
            seen_norm_local = set()
            for name in audio_list:
                try:
                    if not name:
                        continue
                    norm = unicodedata.normalize('NFKC', name).strip().lower()
                    norm = re.sub(r"[^\w\s]", "", norm)
                    norm = re.sub(r"\s+", " ", norm).strip()
                    if norm and norm not in seen_norm_local:
                        unique_audio.append(name)
                        seen_norm_local.add(norm)
                except Exception:
                    continue
            self.audio_combo.addItems(unique_audio)
        else:
            self.audio_combo.addItem("No Audio Input Devices")
        idx = -1
        try:
            idx = self.audio_combo.findText(cur_audio)
        except Exception:
            idx = -1
        if idx != -1:
            self.audio_combo.setCurrentIndex(idx)
        else:
            self.audio_combo.setCurrentIndex(0)
        self.audio_combo.blockSignals(False)

        self.video_devices = video_list
        # store deduplicated audio devices
        try:
            self.audio_devices = unique_audio
        except Exception:
            self.audio_devices = audio_list

    def open_settings(self):
        dialog = SettingsDialog(self)
        dialog.setStyleSheet(apply_theme())
        if dialog.exec() == QtWidgets.QDialog.DialogCode.Accepted:
            new_settings = dialog.get_settings()
            SETTINGS["theme"] = new_settings["theme"]
            SETTINGS["language"] = new_settings["language"]
            SETTINGS["resolution"] = new_settings["resolution"]
            SETTINGS["fps"] = new_settings["fps"]
            save_settings(SETTINGS)
            # Apply new settings
            style = apply_theme()
            self.setStyleSheet(style)
            self.fullscreen_window.setStyleSheet(style)
            self.update_ui_text()
            # Restart camera with new settings
            res = new_settings["resolution"]
            fps = new_settings["fps"]
            cam_idx = self.camera_combo.currentIndex()
            self._start_camera(index=cam_idx, resolution=tuple(res), fps=fps)

    def update_ui_text(self):
        self.camera_label.setText(get_text("camera"))
        self.audio_label.setText(get_text("audio_in"))
        self.debug_btn.setText(get_text("toggle_debug"))
        self.fullscreen_btn.setText(get_text("fullscreen"))
        self.settings_btn.setText(get_text("settings"))

    def _compose_debug_text(self):
        return f"Video FPS: {self.video_fps:.1f} | CPU: {self.cpu:.1f}% | RAM: {self.ram_mb:.1f} MB"

    def _update_stats_and_overlay(self):
        try:
            self.cpu = psutil.cpu_percent(interval=None)
            mem = psutil.Process().memory_info()
            self.ram_mb = mem.rss / (1024*1024)
        except Exception:
            pass

        if self.debug_mode:
            text = self._compose_debug_text()
            self.debug_label.setText(text)
            self.debug_label.adjustSize()
            self.debug_label.show()
            self.fullscreen_window.set_debug_text(text)
        else:
            self.debug_label.hide()
            self.fullscreen_window.debug_label.hide()

    def closeEvent(self, event):
        try:
            if self.camera_thread:
                self.camera_thread.stop()
        except Exception:
            pass
        try:
            if self.audio_worker:
                self.audio_worker.stop()
        except Exception:
            pass
        try:
            if hasattr(self, "_device_poller_stop"):
                self._device_poller_stop.set()
        except Exception:
            pass
        super().closeEvent(event)

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
