# --- VIDEO PLAYER CLASS ---
class DualVideoPlayer:
    def __init__(self, master, video1_path, video2_path):
        self.master = master
        self.master.title("Video Comparison")

        self.cap1 = cv2.VideoCapture(video1_path)
        self.cap2 = cv2.VideoCapture(video2_path)

        self.playing = False

        self.canvas1 = tk.Label(master)
        self.canvas1.pack(side=tk.LEFT)

        self.canvas2 = tk.Label(master)
        self.canvas2.pack(side=tk.LEFT)

        controls = tk.Frame(master)
        controls.pack(side=tk.BOTTOM, fill=tk.X)

        ttk.Button(controls, text="Play", command=self.play).pack(side=tk.LEFT)
        ttk.Button(controls, text="Pause", command=self.pause).pack(side=tk.LEFT)
        ttk.Button(controls, text="Rewind", command=self.rewind).pack(side=tk.LEFT)

    def play(self):
        if not self.playing:
            self.playing = True
            threading.Thread(target=self.update).start()

    def pause(self):
        self.playing = False

    def rewind(self):
        self.cap1.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.cap2.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def update(self):
        while self.playing:
            ret1, frame1 = self.cap1.read()
            ret2, frame2 = self.cap2.read()

            if not ret1 or not ret2:
                self.playing = False
                break

            # Resize and convert color
            frame1 = cv2.resize(frame1, (400, 300))
            frame2 = cv2.resize(frame2, (400, 300))
            frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
            frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

            img1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGBA)
            img2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGBA)

            img1 = tk.PhotoImage(master=self.canvas1, data=cv2.imencode('.png', frame1)[1].tobytes())
            img2 = tk.PhotoImage(master=self.canvas2, data=cv2.imencode('.png', frame2)[1].tobytes())

            self.canvas1.configure(image=img1)
            self.canvas1.image = img1
            self.canvas2.configure(image=img2)
            self.canvas2.image = img2

            time.sleep(1 / 24.0)  # Assuming 24 FPS
