import sys,time,threading,subprocess,psutil,json,re,unicodedata,shutil,os,datetime
import numpy as np,cv2,sounddevice as sd
from PIL import ImageGrab
from pygrabber.dshow_graph import FilterGraph
from PyQt6 import QtCore,QtGui,QtWidgets
try:
    import win32gui,win32con
    WIN32=True
except Exception:
    WIN32=False

SETTINGS={}
try:
    with open('settings.json','r',encoding='utf-8') as f: SETTINGS.update(json.load(f))
except Exception:
    SETTINGS.setdefault('resolution',[1280,720]);SETTINGS.setdefault('fps',120)

T={
 'uk':{'camera':'Камера','resolution':'Розділення','fps':'FPS','audio_in':'Аудіо вхід','toggle_debug':'Дебаг режим','fullscreen':'На весь екран','settings':'Налаштування','settings_title':'Налаштування','theme':'Тема','language':'Мова','dark':'Темна','light':'Світла','ok':'ОК','cancel':'Скасувати','source':'Джерело','android_device':'Android пристрій','sound_on':'Звук увімк.','sound_off':'Звук вимк.'},
 'en':{'camera':'Camera','resolution':'Resolution','fps':'FPS','audio_in':'Audio In','toggle_debug':'Toggle Debug','fullscreen':'Fullscreen','settings':'Settings','settings_title':'Settings','theme':'Theme','language':'Language','dark':'Dark','light':'Light','ok':'OK','cancel':'Cancel','source':'Source','android_device':'Android Device','sound_on':'Sound On','sound_off':'Sound Off'} }
def txt(k): return T.get(SETTINGS.get('language','uk'),T['uk']).get(k,k)

def _log_scrcpy(msg):
    try:
        fn=os.path.join(os.path.dirname(__file__),'scrcpy_debug.log')
        with open(fn,'a',encoding='utf-8') as f:
            f.write(f"[{datetime.datetime.now().isoformat()}] {msg}\n")
    except Exception:
        pass

def apply_theme():
    theme=SETTINGS.get('theme','dark')
    if theme=='dark':
        return """
        QWidget { background-color: #1e1e1e; color: white; }
        QComboBox { background-color: #2e2e2e; color: white; border: 1px solid #555; padding: 3px; }
        QPushButton { background-color: #3e3e3e; color: white; border: 1px solid #555; padding: 5px; }
        QLabel { color: white; }
        """
    else:
        return """
        QWidget { background-color: #ffffff; color: black; }
        QComboBox { background-color: #f5f5f5; color: black; border: 1px solid #ccc; padding: 3px; }
        QPushButton { background-color: #e0e0e0; color: black; border: 1px solid #999; padding: 5px; }
        QLabel { color: black; }
        """

def list_video_devices():
    try:
        g=FilterGraph();d=g.get_input_devices();return d if d else []
    except Exception:
        return []

def list_audio_input_devices():
    try:
        devs=sd.query_devices();out=[];seen=set()
        for d in devs:
            if d.get('max_input_channels',0)>0:
                n=d.get('name');
                if not n: continue
                s=unicodedata.normalize('NFKC',n);s=re.sub(r"[^\w\s]","",s).strip().lower();s=re.sub(r"\s+"," ",s)
                if s and s not in seen: out.append(n);seen.add(s)
        return out
    except Exception:
        return []

class CamThread(QtCore.QThread):
    frame_signal=QtCore.pyqtSignal(np.ndarray);fps_signal=QtCore.pyqtSignal(float)
    def __init__(self,i=0,res=(1280,720),fps=120):
        super().__init__();self.i=i;self.res=res;self.fps=fps;self._run=False;self.cap=None
    def run(self):
        self._run=True
        try:
            self.cap=cv2.VideoCapture(self.i,cv2.CAP_DSHOW)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,int(self.res[0]));self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT,int(self.res[1]));self.cap.set(cv2.CAP_PROP_FPS,int(self.fps));self.cap.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
        except Exception:
            self._run=False;return
        prev=time.time()
        while self._run:
            ret,img=self.cap.read();
            if not ret: time.sleep(0.01); continue
            now=time.time();fps=1.0/(now-prev) if now-prev>1e-6 else 0;prev=now
            self.fps_signal.emit(fps);self.frame_signal.emit(cv2.cvtColor(img,cv2.COLOR_BGR2RGB));time.sleep(0.001)
        try: self.cap.release()
        except Exception: pass
    def stop(self): self._run=False; self.wait(1000)

class AudioWorker(threading.Thread):
    def __init__(self,i=None):
        super().__init__(daemon=True);self.i=i;self._run=threading.Event();self._run.clear();self._muted=False
    def run(self):
        def cb(indata,outdata,frames,time_info,status):
            if status: print('Audio status',status)
            try:
                outdata[:] = 0 if self._muted else indata
            except Exception:
                outdata.fill(0)
        try:
            with sd.Stream(device=(self.i,None),channels=1,samplerate=44100,callback=cb):
                self._run.set();
                while self._run.is_set(): sd.sleep(100)
        except Exception as e: print('Audio error',e)
    def stop(self): self._run.clear()
    def set_muted(self,m): self._muted=bool(m)

class AndroidScreenMirror:
    def __init__(self,did,width=1280,height=720,fps=30):
        self.did=did;self.w=width;self.h=height;self.fps=fps;self.running=False;self._t=None;self.cb=None;self.proc=None
    def start(self,cb,parent_hwnd=None):
        self.cb=cb;self.running=True;self._parent= int(parent_hwnd) if parent_hwnd else None;self._t=threading.Thread(target=self._loop,daemon=True);self._t.start();return True
    def _loop(self):
        try:
            title=f'scrcpy_{self.did}';
            cmd=["scrcpy","-s",self.did,"--video-bit-rate=2M","--max-fps","24","--no-audio","--stay-awake"]
            if self._parent: cmd += ["--window-title",title]
            # Log environment and command for troubleshooting when running
            # as a windowed (no-console) frozen exe.
            try:
                _log_scrcpy(f"PATH={os.environ.get('PATH')}")
                _log_scrcpy(f"which(scrcpy)={shutil.which('scrcpy')}")
                _log_scrcpy(f"cmd={cmd}")
            except Exception:
                pass
            # When running as a windowed (no-console) PyInstaller app, keeping
            # stdout/stderr as pipes can block the child if nobody reads them.
            # Redirect IO to DEVNULL and set Windows startup flags so scrcpy
            # doesn't depend on the parent console.
            startupinfo=None; creationflags=0
            if WIN32:
                try:
                    startupinfo=subprocess.STARTUPINFO()
                    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                    creationflags=subprocess.CREATE_NO_WINDOW
                except Exception:
                    startupinfo=None
            try:
                self.proc=subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL, startupinfo=startupinfo, creationflags=creationflags)
                _log_scrcpy(f"started scrcpy pid={getattr(self.proc,'pid',None)}")
            except Exception as e:
                _log_scrcpy(f"failed to start scrcpy: {e}")
                raise
            time.sleep(4)
            self.embedded=False
            if self._parent and WIN32:
                try:
                    hwnd=None
                    def enum(h,p):
                        nonlocal hwnd
                        try:
                            if title in win32gui.GetWindowText(h): hwnd=h; return False
                        except Exception: pass
                        return True
                    win32gui.EnumWindows(enum,None)
                    if hwnd:
                        try:
                            win32gui.SetParent(hwnd,self._parent)
                            l,t,r,b=win32gui.GetClientRect(self._parent);win32gui.SetWindowPos(hwnd,0,0,0,r-l,b-t,win32con.SWP_NOZORDER)
                            self.embedded=True
                        except Exception as e:
                            print('embed err',e)
                except Exception as e: print('embed err',e)
            if self.proc.poll() is not None:
                _log_scrcpy('scrcpy exited early'); return
            # If embedded successfully, let scrcpy render into our widget and don't screenshot
            if self.embedded:
                while self.running and self.proc.poll() is None:
                    time.sleep(0.1)
                return
            interval=1.0/self.fps;last=time.time()
            while self.running and self.proc.poll() is None:
                now=time.time();
                if now-last<interval: time.sleep(0.001); continue
                try:
                    img=ImageGrab.grab(bbox=(0,0,self.w,self.h))
                except Exception as e:
                    print('grab err',e); break
                f=np.array(img,dtype=np.uint8)
                if self.cb: self.cb(f)
                last=now
        except Exception as e:
            print('mirror loop err',e)
        finally:
            self.running=False; self._cleanup()
    def _cleanup(self):
        if self.proc:
            try: self.proc.terminate(); self.proc.wait(2)
            except: 
                try: self.proc.kill()
                except: pass
    def stop(self): self.running=False; self._cleanup();
    @staticmethod
    def available():
        try:
            which=shutil.which('scrcpy')
            _log_scrcpy(f"available check which(scrcpy)={which}")
            if which:
                return True
            # final check: try running --version without capturing into pipes
            subprocess.run(["scrcpy","--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2)
            return True
        except Exception as e:
            _log_scrcpy(f"available check failed: {e}")
            return False

class ScrcpyThread(QtCore.QThread):
    frame_signal=QtCore.pyqtSignal(np.ndarray);fps_signal=QtCore.pyqtSignal(float);status_signal=QtCore.pyqtSignal(str)
    def __init__(self,did=None,res=(1280,720),fps=30,parent_hwnd=None):
        super().__init__();self.did=did;self.res=res;self.fps=fps;self.running=False;self.mirror=None;self.parent_hwnd=parent_hwnd;self._cnt=0;self._t0=time.time()
    def run(self):
        self.running=True
        if not AndroidScreenMirror.available(): self.status_signal.emit('scrcpy not found'); self.running=False; return
        self.mirror=AndroidScreenMirror(self.did,width=int(self.res[0]),height=int(self.res[1]),fps=self.fps)
        if not self.mirror.start(self._on_frame,parent_hwnd=self.parent_hwnd): self.status_signal.emit('failed'); self.running=False; return
        self.status_signal.emit('active')
        while self.running: time.sleep(0.1)
    def _on_frame(self,f):
        if not self.running: return
        self._cnt+=1;now=time.time();dt=now-self._t0
        if dt>=1.0: self.fps_signal.emit(self._cnt/dt); self._cnt=0; self._t0=now
        self.frame_signal.emit(f)
    def stop(self):
        self.running=False
        try:
            # request thread quit and wait briefly
            try:
                self.quit()
            except Exception:
                pass
            try:
                self.wait(1000)
            except Exception:
                pass
        except Exception:
            pass
    def mute_audio(self):
        try: subprocess.run(["adb","-s",self.did,"shell","input","keyevent","164"],timeout=5); return True
        except Exception: return False
    def unmute_audio(self): return self.mute_audio()

def save_settings(s,fn='settings.json'):
    try:
        with open(fn,'w',encoding='utf-8') as f: json.dump(s,f,indent=2,ensure_ascii=False)
    except Exception as e:
        print('save settings err',e)

class SettingsDialog(QtWidgets.QDialog):
    def __init__(self,parent=None):
        super().__init__(parent);self.setWindowTitle(txt('settings_title'));self.setModal(True);self.setGeometry(100,100,420,260)
        lay=QtWidgets.QVBoxLayout(self)
        # theme/language
        h=QtWidgets.QHBoxLayout();h.addWidget(QtWidgets.QLabel(txt('theme')));self.theme_combo=QtWidgets.QComboBox();self.theme_combo.addItems([txt('dark'),txt('light')]);h.addWidget(self.theme_combo);lay.addLayout(h)
        h2=QtWidgets.QHBoxLayout();h2.addWidget(QtWidgets.QLabel(txt('language')));self.lang_combo=QtWidgets.QComboBox();self.lang_combo.addItems(['Українська','English']);h2.addWidget(self.lang_combo);lay.addLayout(h2)
        # resolution/fps
        r=QtWidgets.QHBoxLayout();r.addWidget(QtWidgets.QLabel(txt('resolution')));self.res_combo=QtWidgets.QComboBox();self.res_combo.addItems(['640x480','1280x720','1920x1080']);r.addWidget(self.res_combo);lay.addLayout(r)
        f=QtWidgets.QHBoxLayout();f.addWidget(QtWidgets.QLabel(txt('fps')));self.fps_combo=QtWidgets.QComboBox();self.fps_combo.addItems(['10','30','60','120']);f.addWidget(self.fps_combo);lay.addLayout(f)
        # buttons
        b=QtWidgets.QHBoxLayout();ok=QtWidgets.QPushButton(txt('ok'));ok.clicked.connect(self.accept);b.addWidget(ok);b.addWidget(QtWidgets.QPushButton(txt('cancel'),clicked=self.reject));lay.addLayout(b)
    def get(self):
        theme='dark' if 'dark' in self.theme_combo.currentText().lower() else 'light'
        lang='uk' if self.lang_combo.currentIndex()==0 else 'en'
        w,h=map(int,self.res_combo.currentText().split('x'))
        fps=int(self.fps_combo.currentText())
        return {'theme':theme,'language':lang,'resolution':[w,h],'fps':fps}

class FullscreenVideoWindow(QtWidgets.QWidget):
    exit_fullscreen=QtCore.pyqtSignal()
    def __init__(self):
        super().__init__(); self.setWindowFlags(QtCore.Qt.WindowType.FramelessWindowHint);self.video_label=QtWidgets.QLabel(self);self.video_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter);lay=QtWidgets.QVBoxLayout(self);lay.setContentsMargins(0,0,0,0);lay.addWidget(self.video_label)
        self.debug_label=QtWidgets.QLabel(self);self.debug_label.setStyleSheet('color:white;');self.debug_label.hide();
    def set_frame(self,frame):
        if frame is None: return
        h,w,ch=frame.shape;bytes_per_line=ch*w;img=QtGui.QImage(frame.data.tobytes(),w,h,bytes_per_line,QtGui.QImage.Format.Format_RGB888);pix=QtGui.QPixmap.fromImage(img).scaled(self.width(),self.height(),QtCore.Qt.AspectRatioMode.IgnoreAspectRatio);self.video_label.setPixmap(pix)
    def set_debug_text(self,t): self.debug_label.setText(t);self.debug_label.adjustSize();self.debug_label.show();
    def showFullScreen(self): super().showFullScreen(); self.video_label.setFixedSize(self.size()); self.debug_label.raise_()
    def keyPressEvent(self,e): self.exit_fullscreen.emit()
    def mousePressEvent(self,e): self.exit_fullscreen.emit()

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__(); self.setWindowTitle('CCard'); self.resize(1280,720)
        self.cur_frame=None;self.fps=0;self.debug=True
        c=QtWidgets.QWidget();self.setCentralWidget(c);l=QtWidgets.QVBoxLayout(c);l.setContentsMargins(0,0,0,0)
        self.video_label=QtWidgets.QLabel();self.video_label.setStyleSheet('background:black');self.video_label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter);l.addWidget(self.video_label,1)
        bar=QtWidgets.QWidget();bar.setFixedHeight(80);hl=QtWidgets.QHBoxLayout(bar);hl.setContentsMargins(6,6,6,6);l.addWidget(bar)
        self.source_combo=QtWidgets.QComboBox();self.source_combo.addItems([txt('camera'),txt('android_device')]);self.source_combo.currentIndexChanged.connect(self.on_source_changed);
        self.source_label=QtWidgets.QLabel(txt('source'));hl.addWidget(self.source_label);hl.addWidget(self.source_combo)
        self.camera_label=QtWidgets.QLabel(txt('camera'));hl.addWidget(self.camera_label)
        self.camera_combo=QtWidgets.QComboBox();self.camera_combo.addItems(list_video_devices() or ['No Devices']);hl.addWidget(self.camera_combo)
        self.android_device_combo=QtWidgets.QComboBox();self.android_device_combo.hide();hl.addWidget(self.android_device_combo)
        self.refresh_android_btn=QtWidgets.QPushButton('Scan');self.refresh_android_btn.hide();self.refresh_android_btn.clicked.connect(self.refresh_android_devices);hl.addWidget(self.refresh_android_btn)
        hl.addStretch();
        self.audio_label=QtWidgets.QLabel(txt('audio_in'));hl.addWidget(self.audio_label)
        self.audio_combo=QtWidgets.QComboBox();self.audio_combo.addItems(list_audio_input_devices() or ['No Audio']);hl.addWidget(self.audio_combo)
        self.debug_btn=QtWidgets.QPushButton(txt('toggle_debug'));hl.addWidget(self.debug_btn);self.debug_btn.clicked.connect(self._toggle_debug)
        self.fullscreen_btn=QtWidgets.QPushButton(txt('fullscreen'));hl.addWidget(self.fullscreen_btn);self.fullscreen_btn.clicked.connect(self._open_fullscreen)
        self.settings_btn=QtWidgets.QPushButton(txt('settings'));hl.addWidget(self.settings_btn);self.settings_btn.clicked.connect(self.open_settings)
        self.audio_btn=QtWidgets.QPushButton(txt('sound_on'));hl.addWidget(self.audio_btn);self.audio_btn.hide();self.audio_btn.clicked.connect(self.toggle_audio)
        self.debug_label=QtWidgets.QLabel(self.video_label);self.debug_label.setStyleSheet('color:white;background:transparent');self.debug_label.move(10,10);self.debug_label.hide()
        self.cam_thread=None;self.scrcpy_thread=None;self.audio_worker=None
        self.setMinimumSize(400,300)
        self._start_camera(0,tuple(SETTINGS.get('resolution',[1280,720])),SETTINGS.get('fps',120))
        self._start_audio_default()
        self.audio_combo.currentIndexChanged.connect(self.on_audio_changed)
        self.ui_timer=QtCore.QTimer(self);self.ui_timer.setInterval(500);self.ui_timer.timeout.connect(self._update_stats_and_overlay);self.ui_timer.start()
        self.fullscreen_window=FullscreenVideoWindow();self.fullscreen_window.exit_fullscreen.connect(self._close_fullscreen)
        self.is_debug=True

    def _start_camera(self,idx,res,fps):
        try:
            if self.cam_thread and self.cam_thread.isRunning(): self.cam_thread.stop()
        except: pass
        self.cam_thread=CamThread(i=idx,res=res,fps=fps);self.cam_thread.frame_signal.connect(self._on_frame);self.cam_thread.fps_signal.connect(self._on_fps);self.cam_thread.start()

    def _start_scrcpy(self,did=None,res=(1280,720),fps=30):
        try:
            if self.scrcpy_thread and self.scrcpy_thread.isRunning(): self.scrcpy_thread.stop()
        except: pass
        # prefer explicit did, otherwise take selected android device when in Android mode
        sel = None
        if did:
            sel = did
        elif self.source_combo.currentIndex()==1 and self.android_device_combo.count()>0:
            sel = self.android_device_combo.currentText()
        else:
            sel = self.camera_combo.currentText()
        # Validate selected device (avoid placeholder values)
        if not sel or any(x in str(sel) for x in ('No Devices','No Devices Found')):
            QtWidgets.QMessageBox.warning(None, 'Error', 'No Android device selected. Scan first!')
            return
        dev = sel
        self.scrcpy_thread=ScrcpyThread(did=dev,res=res,fps=fps,parent_hwnd=int(self.video_label.winId()) if hasattr(self.video_label,'winId') else None)
        self.scrcpy_thread.frame_signal.connect(self._on_frame);self.scrcpy_thread.fps_signal.connect(self._on_fps);self.scrcpy_thread.status_signal.connect(lambda s:print('scrcpy',s));self.scrcpy_thread.start()

    def _on_frame(self,frame):
        self.cur_frame=frame
        # If Android source selected, show prompt unless scrcpy is embedded
        if self.source_combo.currentIndex()==1:
            try:
                if self.scrcpy_thread and getattr(self.scrcpy_thread,'mirror',None) and getattr(self.scrcpy_thread.mirror,'embedded',False):
                    return
            except Exception:
                pass
            self._show_prompt(); return
        if frame is None: return
        h,w,ch=frame.shape;bytes_per_line=ch*w
        q=QtGui.QImage(frame.data.tobytes(),w,h,bytes_per_line,QtGui.QImage.Format.Format_RGB888)
        pix=QtGui.QPixmap.fromImage(q).scaled(self.video_label.width(),self.video_label.height(),QtCore.Qt.AspectRatioMode.IgnoreAspectRatio)
        self.video_label.setPixmap(pix)

    def _show_prompt(self):
        w=self.video_label.width() or 640;h=self.video_label.height() or 480
        p=QtGui.QPixmap(w,h);p.fill(QtGui.QColor('black'));pt=QtGui.QPainter(p);pt.setPen(QtGui.QColor('white'));f=pt.font();f.setPointSize(20);pt.setFont(f);pt.drawText(p.rect(),QtCore.Qt.AlignmentFlag.AlignCenter,'Switch to phone window');pt.end();self.video_label.setPixmap(p)

    def _on_fps(self,f): self.fps=f

    def _start_audio_default(self):
        lst=list_audio_input_devices()
        if not lst: return
        try:
            devs=sd.query_devices()
            for i,d in enumerate(devs):
                if d['max_input_channels']>0:
                    sd.default.device=(i,None); self._start_audio(i); break
        except Exception as e: print('audio start err',e)

    def refresh_android_devices(self):
        try:
            out=subprocess.run(['adb','devices'],capture_output=True,text=True,timeout=5)
            lines=[l.strip() for l in out.stdout.splitlines() if l.strip() and not l.startswith('List of devices')]
            devs=[]
            for l in lines:
                parts=l.split()# '6e61342d0306\tdevice'
                if parts:
                    devs.append(parts[0])
            if not devs:
                devs=[]
            self.android_device_combo.blockSignals(True);
            self.android_device_combo.clear();
            if devs:
                self.android_device_combo.addItems(devs)
            else:
                self.android_device_combo.addItem('No Devices Found')
            self.android_device_combo.blockSignals(False)
            self.refresh_android_btn.setText('Refresh ✓' if devs else 'Refresh')
        except Exception as e:
            print('adb scan err',e); self.android_device_combo.clear(); self.android_device_combo.addItem('No Devices Found')

    def _start_audio(self,i):
        try:
            if self.audio_worker: self.audio_worker.stop()
        except: pass
        self.audio_worker=AudioWorker(i); self.audio_worker.start()

    def on_audio_changed(self,idx):
        name=self.audio_combo.currentText()
        try:
            devices=sd.query_devices()
            found=None
            for i,d in enumerate(devices):
                if d.get('name')==name and d.get('max_input_channels',0)>0:
                    found=i;break
            if found is not None:
                sd.default.device=(found,None)
                self._start_audio(found)
        except Exception as e:
            print('on_audio_changed err',e)

    def open_settings(self):
        dlg=SettingsDialog(self);dlg.theme_combo.setCurrentText(txt(SETTINGS.get('theme','dark')))
        dlg.lang_combo.setCurrentIndex(0 if SETTINGS.get('language','uk')=='uk' else 1)
        res=f"{SETTINGS.get('resolution',[1280,720])[0]}x{SETTINGS.get('resolution',[1280,720])[1]}";dlg.res_combo.setCurrentText(res)
        dlg.fps_combo.setCurrentText(str(SETTINGS.get('fps',120)))
        if dlg.exec()==QtWidgets.QDialog.DialogCode.Accepted:
            new=dlg.get(); SETTINGS.update(new); save_settings(SETTINGS)
            # apply theme/text
            QtWidgets.QApplication.instance().setStyleSheet(apply_theme())
            self.fullscreen_window.setStyleSheet(apply_theme())
            self.update_ui_text()
            # restart camera
            self._start_camera(self.camera_combo.currentIndex(),tuple(SETTINGS.get('resolution',[1280,720])),SETTINGS.get('fps',120))

    def update_ui_text(self):
        self.debug_btn.setText(txt('toggle_debug'));self.fullscreen_btn.setText(txt('fullscreen'));self.settings_btn.setText(txt('settings'))
        # update labels
        try:
            self.source_label.setText(txt('source'))
            # update source combo displayed names but keep current index
            idx=self.source_combo.currentIndex()
            self.source_combo.blockSignals(True)
            self.source_combo.clear(); self.source_combo.addItems([txt('camera'), txt('android_device')])
            self.source_combo.setCurrentIndex(idx)
            self.camera_label.setText(txt('camera'))
            self.audio_label.setText(txt('audio_in'))
        except Exception:
            pass

    def _compose_debug_text(self):
        cpu=psutil.cpu_percent(interval=None)
        mem=psutil.Process().memory_info().rss/1024/1024
        return f"Video FPS: {self.fps:.1f} | CPU: {cpu:.1f}% | RAM: {mem:.1f} MB"

    def _update_stats_and_overlay(self):
        if self.is_debug:
            self.debug_label.setText(self._compose_debug_text());self.debug_label.adjustSize();self.debug_label.show();
            if self.fullscreen_window.isVisible(): self.fullscreen_window.set_debug_text(self._compose_debug_text())
        else:
            self.debug_label.hide(); self.fullscreen_window.debug_label.hide()

    def _toggle_debug(self): self.is_debug=not self.is_debug
    def _open_fullscreen(self):
        self.fullscreen_window.resize(QtWidgets.QApplication.primaryScreen().size());
        if self.cur_frame is not None: self.fullscreen_window.set_frame(self.cur_frame)
        if self.is_debug: self.fullscreen_window.set_debug_text(self._compose_debug_text())
        self.fullscreen_window.showFullScreen()
    def _close_fullscreen(self): self.fullscreen_window.hide()

    def closeEvent(self,e):
        # stop and wait for threads to finish to avoid QThread destruction warnings
        try:
            if hasattr(self,'cam_thread') and self.cam_thread:
                try: self.cam_thread.stop()
                except Exception: pass
                try: self.cam_thread.wait(1000)
                except Exception: pass
        except Exception:
            pass
        try:
            if hasattr(self,'scrcpy_thread') and self.scrcpy_thread:
                try: self.scrcpy_thread.stop()
                except Exception: pass
                try: self.scrcpy_thread.wait(1000)
                except Exception: pass
        except Exception:
            pass
        try:
            if hasattr(self,'audio_worker') and self.audio_worker:
                try: self.audio_worker.stop()
                except Exception: pass
        except Exception:
            pass
        try:
            if hasattr(self,'ui_timer') and self.ui_timer:
                try: self.ui_timer.stop()
                except Exception: pass
        except Exception:
            pass
        save_settings(SETTINGS)
        super().closeEvent(e)

    def on_source_changed(self,idx):
        try:
            idx = int(idx)
        except Exception:
            idx = self.source_combo.currentIndex()
        s=self.source_combo.currentText()
        if idx==0:
            # Camera selected
            self.audio_btn.hide(); self._start_camera(self.camera_combo.currentIndex(),tuple(SETTINGS.get('resolution',[1280,720])),SETTINGS.get('fps',120))
            try:
                if self.scrcpy_thread and self.scrcpy_thread.isRunning(): self.scrcpy_thread.stop()
            except: pass
            self.android_device_combo.hide(); self.refresh_android_btn.hide()
            # stop scrcpy if running
            try:
                if self.scrcpy_thread and self.scrcpy_thread.isRunning():
                    self.scrcpy_thread.stop()
            except Exception:
                pass
            # show camera/fullscreen/audio controls
            self.camera_label.show(); self.camera_combo.show(); self.fullscreen_btn.show(); self.audio_label.show(); self.audio_combo.show()
        else:
            # Android selected
            self.audio_btn.show(); self._show_prompt(); self.android_device_combo.show(); self.refresh_android_btn.show(); self.refresh_android_devices();
            # stop camera to avoid mixed frames
            try:
                if self.cam_thread and self.cam_thread.isRunning():
                    self.cam_thread.stop()
            except Exception:
                pass
            # hide camera/fullscreen/audio controls
            self.camera_label.hide(); self.camera_combo.hide(); self.fullscreen_btn.hide(); self.audio_label.hide(); self.audio_combo.hide()
            self._start_scrcpy(res=tuple(SETTINGS.get('resolution',[1280,720])),fps=30)

    def toggle_audio(self):
        state = getattr(self,'audio_on',True); state = not state; self.audio_on=state
        if self.audio_worker: self.audio_worker.set_muted(not state); self.audio_btn.setText(txt('sound_on') if state else txt('sound_off'))
        else:
            if self.scrcpy_thread and self.scrcpy_thread.isRunning():
                if state: self.scrcpy_thread.unmute_audio()
                else: self.scrcpy_thread.mute_audio(); self.audio_btn.setText(txt('sound_on') if state else txt('sound_off'))

def main():
    app=QtWidgets.QApplication(sys.argv);w=MainWindow();w.show();sys.exit(app.exec())

if __name__=='__main__': main()
