from pathlib import Path

class Logger:
    def __init__(self, flush_every=50):
        self.sublogs = []
        self.iters = []
        self.img_losses = []
        self.smooth_losses = [] 
        self.straight_losses = [] 
        self.axis_align_losses = [] 
        self.curvature_losses = [] 
        self.losses = []

        self._buffer = []          # hold log lines in memory
        self._flush_every = flush_every
        self.log_path = None

    def create_log(self, log_path):
        self.log_path = log_path
        # make sure file exists (truncate old content)
        with open(self.log_path, 'w'):
            pass

    def print(self, text):
        """Add log text to buffer. Flush periodically."""
        if not text.endswith("\n"):
            text += "\n"
        self._buffer.append(text)

        if len(self._buffer) >= self._flush_every:
            self.flush()

    def flush(self):
        """Write buffer to disk."""
        if self.log_path and self._buffer:
            with open(self.log_path, 'a') as f:
                f.writelines(self._buffer)
            self._buffer.clear()

    def close(self):
        """Flush remaining logs when training ends."""
        self.flush()

    def log_loss(self, iter, img_loss, smooth_loss, straight_loss, axis_align_loss, curvature_loss, loss):
        """Save loss values to memory (not flushed)."""
        self.iters.append(iter)
        self.img_losses.append(img_loss)
        self.smooth_losses.append(smooth_loss)
        self.straight_losses.append(straight_loss)
        self.axis_align_losses.append(axis_align_loss)
        self.curvature_losses.append(curvature_loss)
        self.losses.append(loss)

    def plot_losses(self, save_path):
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(self.iters, self.img_losses, label='Image Loss')
        plt.plot(self.iters, self.smooth_losses, label='Smoothness Loss')
        plt.plot(self.iters, self.straight_losses, label='Straightness Loss')
        plt.plot(self.iters, self.axis_align_losses, label='Axis-align Loss')
        plt.plot(self.iters, self.curvature_losses, label='Curvature Loss')
        plt.plot(self.iters, self.losses, label='Total Loss')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(save_path)
        plt.close()




logger = Logger()

