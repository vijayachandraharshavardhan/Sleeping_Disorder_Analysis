import cv2, numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from scipy.fft import rfft, rfftfreq
from moviepy import VideoFileClip
import librosa
import os

def detect_snore(video_path):
    """
    Detect snore from video audio.
    Analyzes audio spectrum for low-frequency snoring characteristics.
    """
    clip = None
    temp_audio = 'temp_audio.wav'
    try:
        # Extract audio from video
        clip = VideoFileClip(video_path)
        audio = clip.audio
        if audio is None:
            print("No audio track in video")
            return False
        # Write audio to temp file
        audio.write_audiofile(temp_audio)
        # Load audio
        y, sr = librosa.load(temp_audio, sr=None)
        print(f"Audio length: {len(y)}, sample rate: {sr}")
        if len(y) == 0:
            print("No audio data loaded")
            return False
        # Check audio volume (RMS)
        rms = np.sqrt(np.mean(y**2))
        print(f"Audio RMS: {rms:.4f}")
        if rms < 0.01:  # Too quiet, no snore
            print("No snore detected: audio too quiet")
            return False
        # Analyze audio spectrum
        # Compute FFT
        fft = np.fft.rfft(y)
        freqs = np.fft.rfftfreq(len(y), 1/sr)
        # Focus on low frequencies (snoring is typically 20-300 Hz)
        low_freq_mask = (freqs >= 20) & (freqs <= 300)
        if not np.any(low_freq_mask):
            print("No low frequency data")
            return False
        low_freq_power = np.sum(np.abs(fft[low_freq_mask])**2)
        total_power = np.sum(np.abs(fft)**2)
        if total_power == 0:
            print("No audio power")
            return False
        # Calculate percentage of power in low frequencies
        low_freq_ratio = low_freq_power / total_power
        print(f"Low frequency ratio: {low_freq_ratio:.4f}")
        # Threshold for snore detection (adjust as needed)
        if low_freq_ratio > 0.1:  # If more than 10% of power is in low frequencies
            print("Snore detected: high low-frequency content")
            return True
        else:
            print("No snore detected: low low-frequency content")
            return False
    except Exception as e:
        print(f"Error detecting snore: {e}")
        return False
    finally:
        if clip is not None:
            clip.close()  # Ensure clip is closed to release the video file
        try:
            if os.path.exists(temp_audio):
                os.remove(temp_audio)
        except OSError:
            pass

def bandpass_filter(signal, fs, low=0.5, high=5.0, order=3):
    nyq=0.5*fs
    low_norm = max(0.01, min(low / nyq, 0.99))
    high_norm = max(low_norm + 0.01, min(high / nyq, 0.99))
    b,a=butter(order,[low_norm, high_norm],btype='band')
    return filtfilt(b,a,signal)

def estimate_bpm_from_video(path, duration_limit=None):
    try:
        cap=cv2.VideoCapture(path)
        fps=cap.get(cv2.CAP_PROP_FPS) or 30
        greens=[]; times=[]; brightnesses=[]; i=0
        while True:
            r,f=cap.read()
            if not r: break
            h,w=f.shape[:2]
            cx,cy=w//2,h//2
            crop=f[cy-50:cy+50,cx-50:cx+50] if h>100 and w>100 else f
            greens.append(np.mean(crop[:,:,1]))
            # Calculate average brightness (grayscale)
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            brightnesses.append(brightness)
            times.append(i/fps); i+=1
            if duration_limit and i/fps>=duration_limit: break
        cap.release()
        if not greens or not times:
            return {'bpm_peaks': None, 'bpm_fft': None, 'snore_detected': False}

        s=np.array(greens); t=np.array(times)
        s=s-np.polyval(np.polyfit(t,s,1),t)
        filt=bandpass_filter(s,fps)
        print(f"Filtered signal std: {np.std(filt)}, range: {np.max(filt) - np.min(filt)}")
        print(f"Number of frames: {len(greens)}, fps: {fps}")
        env=np.abs(filt)
        win=max(1,int(fps*0.2))
        smooth=np.convolve(env,np.ones(win)/win,mode='same')
        pk,_=find_peaks(smooth,distance=fps*0.4)
        if len(pk)>=2:
            bpm_peaks=60/np.mean(np.diff(t[pk]))
            print(f"Detected {len(pk)} peaks, BPM from peaks: {bpm_peaks}")
        else:
            bpm_peaks=None
            print(f"Less than 2 peaks detected: {len(pk)}")
        N=len(filt)
        yf=rfft(filt)
        xf=rfftfreq(N,1/fps)
        m=(xf>=0.5)&(xf<=5)
        bpm_fft=None
        if np.any(m):
            bpm_fft=xf[m][np.argmax(np.abs(yf[m]))]*60
        print(f"FFT bpm: {bpm_fft}")
        snore_detected = detect_snore(path)
        return {'bpm_peaks':bpm_peaks,'bpm_fft':bpm_fft, 'snore_detected': snore_detected}
    except Exception as e:
        print(f"Error estimating BPM from video: {e}")
        return {'bpm_peaks': None, 'bpm_fft': None, 'snore_detected': False}
